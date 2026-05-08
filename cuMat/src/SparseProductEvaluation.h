#ifndef __CUMAT_SPARSE_PRODUCT_EVALUATION__
#define __CUMAT_SPARSE_PRODUCT_EVALUATION__

#include "Macros.h"
#include "ProductOp.h"
#include "SparseMatrix.h"

CUMAT_NAMESPACE_BEGIN

namespace internal
{

    // CwiseSrcTag * CwiseSrcTag -> SparseDstTag; outer product
    //This handles all dense cwise+matrix inputs and dense matrix output
    //The sparse methods (SparseSrcTag, SparseDstTag) are handled seperately
    template<
        typename _Dst, ProductArgOp _DstOp,
        typename _SrcLeft, ProductArgOp _SrcLeftOp,
        typename _SrcRight, ProductArgOp _SrcRightOp,
        AssignmentMode _AssignmentMode
    >
    struct ProductAssignment<_Dst, SparseDstTag, _DstOp, _SrcLeft, CwiseSrcTag, _SrcLeftOp, _SrcRight, CwiseSrcTag, _SrcRightOp, _AssignmentMode>
    {
        using Op = ProductOp<_SrcLeft, _SrcRight, _SrcLeftOp, _SrcRightOp, _DstOp>;
        using Scalar = typename Op::Scalar;

        static void assign(_Dst& dst, const Op& op) {
            //Check that the input matrices are vectors
            CUMAT_STATIC_ASSERT((Op::TransposedLeft ? Op::RowsLeft : Op::ColumnsLeft == 1),
                "Product evaluation into a sparse matrix is only supported for the outer product of two vectors, left matrix is not a column vector");
            CUMAT_STATIC_ASSERT((Op::TransposedRight ? Op::ColumnsRight : Op::RowsRight == 1),
                "Product evaluation into a sparse matrix is only supported for the outer product of two vectors, right matrix is not a row vector");

            //launch cwise-evaluation
            Assignment<_Dst, Op, _AssignmentMode, typename traits<_Dst>::DstTag, CwiseSrcTag>::assign(dst, op);
        }
    };

#if CUMAT_NVCC==1
    namespace kernels
    {

    template <typename L, typename R, typename M, AssignmentMode Mode, int Batches,
        bool BroadcastMatrix = internal::traits<L>::BatchesAtCompileTime==1,
        bool BroadcastRhs = internal::traits<R>::BatchesAtCompileTime == 1>
    __global__ void CSRMVKernel_StaticBatches(dim3 virtual_size, const L matrix, const R vector, M output)
    {
        typedef typename L::Scalar LeftScalar;
        typedef typename R::Scalar RightScalar;
        typedef typename M::Scalar OutputScalar;
        typedef ProductElementFunctor<LeftScalar, RightScalar, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE> Functor;
		SparsityPattern<CSR>::IndexVector JA = matrix.getSparsityPattern().JA;
		SparsityPattern<CSR>::IndexVector IA = matrix.getSparsityPattern().IA;
		const int nnz = matrix.getSparsityPattern().nnz;
        CUMAT_KERNEL_1D_LOOP(outer, virtual_size)
            int start = JA.getRawCoeff(outer);
            int end = JA.getRawCoeff(outer + 1);
            if (start>=end) continue;
            int inner = IA.getRawCoeff(start);
            OutputScalar value[Batches];
#pragma unroll
            for (int b = 0; b < Batches; ++b) {
                LeftScalar tmp1 = BroadcastMatrix
                    ? matrix.getSparseCoeff(outer, inner, 0, start)
                    : matrix.getSparseCoeff(outer, inner, b, start + b * nnz);
                RightScalar tmp2 = vector.coeff(inner, 0, BroadcastRhs ? 0 : b, -1);
                OutputScalar tmp3 = Functor::mult(tmp1, tmp2);
                value[b] = tmp3;
            }
            for (int i=start+1; i<end; ++i)
            {
                inner = IA.getRawCoeff(i);
#pragma unroll
                for (int b = 0; b < Batches; ++b) {
                    LeftScalar tmp1 = BroadcastMatrix
                        ? matrix.getSparseCoeff(outer, inner, 0, i)
                        : matrix.getSparseCoeff(outer, inner, b, i + b * nnz);
                    RightScalar tmp2 = vector.coeff(inner, 0, BroadcastRhs ? 0 : b, -1);
                    OutputScalar tmp3 = Functor::mult(tmp1, tmp2);
                    value[b] += tmp3;
                }
            }
#pragma unroll
            for (int b = 0; b < Batches; ++b) {
                internal::CwiseAssignmentHandler<M, OutputScalar, Mode>::assign(output, value[b], outer + b*output.rows());
            }
		CUMAT_KERNEL_1D_LOOP_END
    }

    //CSR SpMM kernel: sparse matrix * dense matrix -> dense matrix
    //One thread per (row, col) of the output
    template <typename L, typename R, typename M, AssignmentMode Mode, int Batches,
        bool BroadcastMatrix = internal::traits<L>::BatchesAtCompileTime==1,
        bool BroadcastRhs = internal::traits<R>::BatchesAtCompileTime == 1>
    __global__ void CSRMMKernel_StaticBatches(dim3 virtual_size, const L matrix, const R dense, M output)
    {
        typedef typename L::Scalar LeftScalar;
        typedef typename R::Scalar RightScalar;
        typedef typename M::Scalar OutputScalar;
        typedef ProductElementFunctor<LeftScalar, RightScalar, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE> Functor;
        SparsityPattern<CSR>::IndexVector JA = matrix.getSparsityPattern().JA;
        SparsityPattern<CSR>::IndexVector IA = matrix.getSparsityPattern().IA;
        const int nnz = matrix.getSparsityPattern().nnz;
        CUMAT_KERNEL_2D_LOOP(row, col, virtual_size)
            int start = JA.getRawCoeff(row);
            int end = JA.getRawCoeff(row + 1);
            OutputScalar value[Batches];
#pragma unroll
            for (int b = 0; b < Batches; ++b) value[b] = OutputScalar(0);
            for (int i = start; i < end; ++i)
            {
                int inner = IA.getRawCoeff(i);
#pragma unroll
                for (int b = 0; b < Batches; ++b)
                {
                    LeftScalar tmp1 = BroadcastMatrix
                        ? matrix.getSparseCoeff(row, inner, 0, i)
                        : matrix.getSparseCoeff(row, inner, b, i + b * nnz);
                    RightScalar tmp2 = dense.coeff(inner, col, BroadcastRhs ? 0 : b, -1);
                    OutputScalar tmp3 = Functor::mult(tmp1, tmp2);
                    value[b] += tmp3;
                }
            }
#pragma unroll
            for (int b = 0; b < Batches; ++b) {
                Index linearIndex;
                if (CUMAT_IS_COLUMN_MAJOR(traits<M>::Flags))
                    linearIndex = row + col * output.rows();
                else
                    linearIndex = row * output.cols() + col;
                linearIndex += b * output.rows() * output.cols();
                internal::CwiseAssignmentHandler<M, OutputScalar, Mode>::assign(output, value[b], linearIndex);
            }
        CUMAT_KERNEL_2D_LOOP_END
    }

    //CSC SpMV kernel: one thread per column, uses atomicAdd for output accumulation
    template <typename L, typename R, typename M, AssignmentMode Mode, int Batches,
        bool BroadcastMatrix = internal::traits<L>::BatchesAtCompileTime==1,
        bool BroadcastRhs = internal::traits<R>::BatchesAtCompileTime == 1>
    __global__ void CSCMVKernel_StaticBatches(dim3 virtual_size, const L matrix, const R vector, M output)
    {
        typedef typename L::Scalar LeftScalar;
        typedef typename R::Scalar RightScalar;
        typedef typename M::Scalar OutputScalar;
        typedef ProductElementFunctor<LeftScalar, RightScalar, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE> Functor;
        SparsityPattern<CSC>::IndexVector JA = matrix.getSparsityPattern().JA;
        SparsityPattern<CSC>::IndexVector IA = matrix.getSparsityPattern().IA;
        const int nnz = matrix.getSparsityPattern().nnz;
        const int rows = matrix.rows();
        OutputScalar* outputData = output.data();
        CUMAT_KERNEL_1D_LOOP(outer, virtual_size)
            int start = JA.getRawCoeff(outer);
            int end = JA.getRawCoeff(outer + 1);
            if (start >= end) continue;
            for (int i = start; i < end; ++i)
            {
                int inner = IA.getRawCoeff(i);
#pragma unroll
                for (int b = 0; b < Batches; ++b) {
                    LeftScalar tmp1 = BroadcastMatrix
                        ? matrix.getSparseCoeff(inner, outer, 0, i)
                        : matrix.getSparseCoeff(inner, outer, b, i + b * nnz);
                    RightScalar tmp2 = vector.coeff(outer, 0, BroadcastRhs ? 0 : b, -1);
                    OutputScalar tmp3 = Functor::mult(tmp1, tmp2);
                    atomicAdd(&outputData[inner + b * rows], tmp3);
                }
            }
        CUMAT_KERNEL_1D_LOOP_END
    }

    }

    //CwiseSrcTag (Sparse) * CwiseSrcTag (Dense) -> DenseDstTag, CSR sparse matrix-dense product
    //Dispatches to SpMV (vector RHS) or SpMM (matrix RHS)
    //CSR SparseMatrix
    //TODO: support vector on the left, transposed and conjugated versions
    template<
        typename _Dst,
        typename _SrcLeftScalar, int _SrcLeftBatches,
        typename _SrcRight,
        AssignmentMode _AssignmentMode
    >
    struct ProductAssignment<
        _Dst, DenseDstTag, ProductArgOp::NONE, 
        SparseMatrix<_SrcLeftScalar, _SrcLeftBatches, SparseFlags::CSR>, CwiseSrcTag, ProductArgOp::NONE,
        _SrcRight, CwiseSrcTag, ProductArgOp::NONE, 
        _AssignmentMode>
    {
        using SrcLeft = SparseMatrix<_SrcLeftScalar, _SrcLeftBatches, SparseFlags::CSR>;
        using Op = ProductOp<SrcLeft, _SrcRight, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE>;
        using Scalar = typename Op::Scalar;

        CUMAT_STATIC_ASSERT((Op::Batches != Dynamic),
                "SparseMatrix - dense product does only support compile-time fixed batch count");

        static void assign(_Dst& dst, const Op& op) {
            
            typedef typename _Dst::Type DstActual;
            CUMAT_PROFILING_INC(EvalMatmulSparse);
            CUMAT_PROFILING_INC(EvalAny);
            if (dst.size() == 0) return;
            CUMAT_ASSERT(op.rows() == dst.rows());
            CUMAT_ASSERT(op.cols() == dst.cols());
            CUMAT_ASSERT(op.batches() == dst.batches());
            CUMAT_ASSERT(op.batches() == Op::Batches);

            Context& ctx = Context::current();
            if (Op::ColumnsRight == 1)
            {
                //SpMV: matrix * vector
                CUMAT_LOG_DEBUG("Evaluate CSR SparseMatrix-DenseVector multiplication " << typeid(op.derived()).name()
                    << " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols());
                KernelLaunchConfig cfg = ctx.createLaunchConfig1D(dst.rows(), kernels::CSRMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
                kernels::CSRMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                    <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
                    (cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
            } else {
                //SpMM: matrix * dense matrix
                CUMAT_LOG_DEBUG("Evaluate CSR SparseMatrix-DenseMatrix multiplication " << typeid(op.derived()).name()
                    << " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols()
                    << ", rhsCols=" << op.right().cols());
                dim3 virtualSize(dst.rows(), dst.cols(), 1);
                KernelLaunchConfig cfg = ctx.createLaunchConfig1D(virtualSize.x * virtualSize.y, kernels::CSRMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
                kernels::CSRMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                    <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
                    (virtualSize, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
            }
            CUMAT_CHECK_ERROR();
            CUMAT_LOG_DEBUG("Evaluation done");
        }
    };

    //CwiseSrcTag (Sparse) * CwiseSrcTag (Dense-Vector) -> DenseDstTag (Vector-Vector), CSC sparse matrix-vector product
    template<
        typename _Dst,
        typename _SrcLeftScalar, int _SrcLeftBatches,
        typename _SrcRight,
        AssignmentMode _AssignmentMode
    >
    struct ProductAssignment<
        _Dst, DenseDstTag, ProductArgOp::NONE, 
        SparseMatrix<_SrcLeftScalar, _SrcLeftBatches, SparseFlags::CSC>, CwiseSrcTag, ProductArgOp::NONE,
        _SrcRight, CwiseSrcTag, ProductArgOp::NONE, 
        _AssignmentMode>
    {
        using SrcLeft = SparseMatrix<_SrcLeftScalar, _SrcLeftBatches, SparseFlags::CSC>;
        using Op = ProductOp<SrcLeft, _SrcRight, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE>;
        using Scalar = typename Op::Scalar;

        CUMAT_STATIC_ASSERT((Op::ColumnsRight == 1),
                "CSC SparseMatrix - DenseVector product only supports column vectors as right argument, use batches instead");
        CUMAT_STATIC_ASSERT((Op::Batches != Dynamic),
                "CSC SparseMatrix - DenseVector does only support compile-time fixed batch count");

        static void assign(_Dst& dst, const Op& op) {
            
            typedef typename _Dst::Type DstActual;
            CUMAT_PROFILING_INC(EvalMatmulSparse);
            CUMAT_PROFILING_INC(EvalAny);
            if (dst.size() == 0) return;
            CUMAT_ASSERT(op.rows() == dst.rows());
            CUMAT_ASSERT(op.cols() == dst.cols());
            CUMAT_ASSERT(op.batches() == dst.batches());
            CUMAT_ASSERT(op.batches() == Op::Batches);

			CUMAT_LOG_DEBUG("Evaluate CSC SparseMatrix-DenseVector multiplication " << typeid(op.derived()).name()
				<< " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols());;

            //here is now the real logic
            Context& ctx = Context::current();
            //CSC SpMV uses one thread per column; output must be zero-initialized for atomicAdd accumulation
            dst.setZero();
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(op.left().cols(), kernels::CSCMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
			kernels::CSCMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
                (cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
            CUMAT_CHECK_ERROR();
            CUMAT_LOG_DEBUG("Evaluation done");
        }
    };

    //CwiseSrcTag (SparseExpressionOp) * CwiseSrcTag (Dense-Vector) -> DenseDstTag (Vector-Vector), sparse matrix-vector product
    //Currently, only non-batched CSR matrices are supported
    //TODO: support also CSC, vector on the left, transposed and conjugated versions
    template<
        typename _Dst,
        typename _SrcLeftChild,
        typename _SrcRight,
        AssignmentMode _AssignmentMode
    >
    struct ProductAssignment<
        _Dst, DenseDstTag, ProductArgOp::NONE, 
        SparseExpressionOp<_SrcLeftChild, SparseFlags::CSR>, CwiseSrcTag, ProductArgOp::NONE, 
        _SrcRight, CwiseSrcTag, ProductArgOp::NONE, 
        _AssignmentMode>
    {
        using SrcLeft = SparseExpressionOp<_SrcLeftChild, SparseFlags::CSR>;
        using Op = ProductOp<SrcLeft, _SrcRight, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE>;
        using Scalar = typename Op::Scalar;

        CUMAT_STATIC_ASSERT((Op::ColumnsRight == 1),
                "SparseMatrix - DenseVector product only supports column vectors as right argument (for now)");
        CUMAT_STATIC_ASSERT((Op::Batches != Dynamic),
            "SparseMatrix - DenseVector does only support compile-time fixed batch count");

        static void assign(_Dst& dst, const Op& op) {
            
            typedef typename _Dst::Type DstActual;
            CUMAT_PROFILING_INC(EvalMatmulSparse);
            CUMAT_PROFILING_INC(EvalAny);
            if (dst.size() == 0) return;
            CUMAT_ASSERT(op.rows() == dst.rows());
            CUMAT_ASSERT(op.cols() == dst.cols());
            CUMAT_ASSERT(op.batches() == dst.batches());

			CUMAT_LOG_DEBUG("Evaluate SparseMatrix-DenseVector multiplication " << typeid(op.derived()).name()
				<< " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols());;

            //here is now the real logic
            Context& ctx = Context::current();
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(static_cast<unsigned int>(dst.size()), kernels::CSRMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
			kernels::CSRMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
                (cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
            CUMAT_CHECK_ERROR();
            CUMAT_LOG_DEBUG("Evaluation done");
        }
    };



	namespace kernels
	{
		//ELLPACK Matrix-Vector kernel. One thread per row
		template <typename L, typename R, typename M, AssignmentMode Mode, int Batches,
			bool BroadcastMatrix = internal::traits<L>::BatchesAtCompileTime == 1,
			bool BroadcastRhs = internal::traits<R>::BatchesAtCompileTime == 1>
			__global__ void ELLPACKMVKernel_StaticBatches(dim3 virtual_size, const L matrix, const R vector, M output)
		{
			typedef typename L::Scalar LeftScalar;
			typedef typename R::Scalar RightScalar;
			typedef typename M::Scalar OutputScalar;
			typedef ProductElementFunctor<LeftScalar, RightScalar, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE> Functor;
			const SparsityPattern<SparseFlags::ELLPACK>::IndexMatrix& indices = matrix.getSparsityPattern().indices;
			const int nnzPerRow = matrix.getSparsityPattern().nnzPerRow;
			const int rows = matrix.getSparsityPattern().rows;
			CUMAT_KERNEL_1D_LOOP(row, virtual_size)
				OutputScalar value[Batches] = {0};
				for (int ci = 0; ci < nnzPerRow; ++ci) {
					int col = indices.coeff(row, ci, 0, -1);
					if (col < 0) continue; //TODO: test if it is faster to continue reading (and set col=0) and discard before assignment
#pragma unroll
					for (int b = 0; b < Batches; ++b) {
						LeftScalar tmp1 = BroadcastMatrix
							? matrix.getSparseCoeff(row, col, 0, row + ci*rows)
							: matrix.getSparseCoeff(row, col, b, row + rows*(ci + b*nnzPerRow));
						RightScalar tmp2 = vector.coeff(col, 0, BroadcastRhs ? 0 : b, -1);
						OutputScalar tmp3 = Functor::mult(tmp1, tmp2);
						value[b] += tmp3;
					}
				}
#pragma unroll
				for (int b = 0; b < Batches; ++b) {
					internal::CwiseAssignmentHandler<M, OutputScalar, Mode>::assign(output, value[b], row + b * output.rows());
				}
			CUMAT_KERNEL_1D_LOOP_END
		}

	}

	//CwiseSrcTag (Sparse) * CwiseSrcTag (Dense-Vector) -> DenseDstTag (Vector-Vector), sparse matrix-vector product
	//ELLPACK
	template<
		typename _Dst,// ProductArgOp _DstOp,
		typename _SrcLeftScalar, int _SrcLeftBatches,// ProductArgOp _SrcLeftOp,
		typename _SrcRight,// ProductArgOp _SrcRightOp,
		AssignmentMode _AssignmentMode
	>
		struct ProductAssignment<
		_Dst, DenseDstTag, ProductArgOp::NONE,
		SparseMatrix<_SrcLeftScalar, _SrcLeftBatches, SparseFlags::ELLPACK>, CwiseSrcTag, ProductArgOp::NONE,
		_SrcRight, CwiseSrcTag, ProductArgOp::NONE,
		_AssignmentMode>
	{
		using SrcLeft = SparseMatrix<_SrcLeftScalar, _SrcLeftBatches, SparseFlags::ELLPACK>;
		using Op = ProductOp<SrcLeft, _SrcRight, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE>;
		using Scalar = typename Op::Scalar;

		CUMAT_STATIC_ASSERT((Op::ColumnsRight == 1),
			"SparseMatrix - DenseVector product only supports column vectors as right argument, use batches instead");
		CUMAT_STATIC_ASSERT((Op::Batches != Dynamic),
			"SparseMatrix - DenseVector does only support compile-time fixed batch count");

		static void assign(_Dst& dst, const Op& op) {

			typedef typename _Dst::Type DstActual;
			CUMAT_PROFILING_INC(EvalMatmulSparse);
			CUMAT_PROFILING_INC(EvalAny);
			if (dst.size() == 0) return;
			CUMAT_ASSERT(op.rows() == dst.rows());
			CUMAT_ASSERT(op.cols() == dst.cols());
			CUMAT_ASSERT(op.batches() == dst.batches());
			CUMAT_ASSERT(op.batches() == Op::Batches);

			CUMAT_LOG_DEBUG("Evaluate SparseMatrix-DenseVector multiplication " << typeid(op.derived()).name()
				<< " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols());;

			//here is now the real logic
			Context& ctx = Context::current();
			KernelLaunchConfig cfg = ctx.createLaunchConfig1D(dst.rows(), kernels::ELLPACKMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
			kernels::ELLPACKMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
				<< <cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >> >
				(cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
			CUMAT_CHECK_ERROR();
			CUMAT_LOG_DEBUG("Evaluation done");
		}
	};

	//CwiseSrcTag (SparseExpressionOp) * CwiseSrcTag (Dense-Vector) -> DenseDstTag (Vector-Vector), sparse matrix-vector product
	//ELLPACK SparseExpressionOp
	//TODO: support vector on the left, transposed and conjugated versions
	template<
		typename _Dst,
		typename _SrcLeftChild,
		typename _SrcRight,
		AssignmentMode _AssignmentMode
	>
		struct ProductAssignment<
		_Dst, DenseDstTag, ProductArgOp::NONE,
		SparseExpressionOp<_SrcLeftChild, SparseFlags::ELLPACK>, CwiseSrcTag, ProductArgOp::NONE,
		_SrcRight, CwiseSrcTag, ProductArgOp::NONE,
		_AssignmentMode>
	{
		using SrcLeft = SparseExpressionOp<_SrcLeftChild, SparseFlags::ELLPACK>;
		using Op = ProductOp<SrcLeft, _SrcRight, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE>;
		using Scalar = typename Op::Scalar;

		CUMAT_STATIC_ASSERT((Op::ColumnsRight == 1),
			"SparseMatrix - DenseVector product only supports column vectors as right argument (for now)");
		CUMAT_STATIC_ASSERT((Op::Batches != Dynamic),
			"SparseMatrix - DenseVector does only support compile-time fixed batch count");

		static void assign(_Dst& dst, const Op& op) {

			typedef typename _Dst::Type DstActual;
			CUMAT_PROFILING_INC(EvalMatmulSparse);
			CUMAT_PROFILING_INC(EvalAny);
			if (dst.size() == 0) return;
			CUMAT_ASSERT(op.rows() == dst.rows());
			CUMAT_ASSERT(op.cols() == dst.cols());
			CUMAT_ASSERT(op.batches() == dst.batches());

			CUMAT_LOG_DEBUG("Evaluate SparseMatrix-DenseVector multiplication " << typeid(op.derived()).name()
				<< " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols());;

			//here is now the real logic
			Context& ctx = Context::current();
			KernelLaunchConfig cfg = ctx.createLaunchConfig1D(static_cast<unsigned int>(dst.size()), kernels::ELLPACKMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
			kernels::ELLPACKMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
				<< <cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >> >
				(cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
			CUMAT_CHECK_ERROR();
			CUMAT_LOG_DEBUG("Evaluation done");
		}
	};

	//CwiseSrcTag (SparseExpressionOp) * CwiseSrcTag (Dense-Vector) -> DenseDstTag (Vector-Vector), CSC sparse matrix-vector product
	template<
		typename _Dst,
		typename _SrcLeftChild,
		typename _SrcRight,
		AssignmentMode _AssignmentMode
	>
		struct ProductAssignment<
		_Dst, DenseDstTag, ProductArgOp::NONE,
		SparseExpressionOp<_SrcLeftChild, SparseFlags::CSC>, CwiseSrcTag, ProductArgOp::NONE,
		_SrcRight, CwiseSrcTag, ProductArgOp::NONE,
		_AssignmentMode>
	{
		using SrcLeft = SparseExpressionOp<_SrcLeftChild, SparseFlags::CSC>;
		using Op = ProductOp<SrcLeft, _SrcRight, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE>;
		using Scalar = typename Op::Scalar;

		CUMAT_STATIC_ASSERT((Op::ColumnsRight == 1),
			"CSC SparseExpressionOp - DenseVector product only supports column vectors as right argument (for now)");
		CUMAT_STATIC_ASSERT((Op::Batches != Dynamic),
			"CSC SparseExpressionOp - DenseVector does only support compile-time fixed batch count");

		static void assign(_Dst& dst, const Op& op) {

			typedef typename _Dst::Type DstActual;
			CUMAT_PROFILING_INC(EvalMatmulSparse);
			CUMAT_PROFILING_INC(EvalAny);
			if (dst.size() == 0) return;
			CUMAT_ASSERT(op.rows() == dst.rows());
			CUMAT_ASSERT(op.cols() == dst.cols());
			CUMAT_ASSERT(op.batches() == dst.batches());

			CUMAT_LOG_DEBUG("Evaluate CSC SparseExpressionOp-DenseVector multiplication " << typeid(op.derived()).name()
				<< " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols());;

			//here is now the real logic
			Context& ctx = Context::current();
			dst.setZero();
			KernelLaunchConfig cfg = ctx.createLaunchConfig1D(op.left().cols(), kernels::CSCMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
			kernels::CSCMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
				<< <cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >> >
				(cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
			CUMAT_CHECK_ERROR();
			CUMAT_LOG_DEBUG("Evaluation done");
		}
	};
#endif
}

CUMAT_NAMESPACE_END

#endif