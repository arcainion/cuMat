#ifndef __CUMAT_SPARSE_PRODUCT_EVALUATION__
#define __CUMAT_SPARSE_PRODUCT_EVALUATION__

#include "Macros.h"
#include "ProductOp.h"
#include "SparseMatrix.h"

#ifndef CUMAT_SPARSE_MM_ACCUM_THRESHOLD
/**
 * \brief Max accumulators per thread in SpMM one-thread-per-row strategy.
 * When cols * Batches <= threshold, one thread per row iterates over all
 * output columns and batches, amortizing JA/IA sparsity-pattern loads
 * across columns. Larger values increase register pressure.
 * Default: 32 (fits 32 floats in registers with headroom for 255-register limit).
 */
#define CUMAT_SPARSE_MM_ACCUM_THRESHOLD 32
#endif

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
    __global__ void __launch_bounds__(256) CSRMVKernel_StaticBatches(dim3 virtual_size, const L matrix, const R vector, M output)
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
    //Uses column-fast work mapping: adjacent threads handle the same row with different columns
    //so column-major dense B access is coalesced.
    //When useOneThreadPerRow is true, one thread per row iterates over all cols and batches,
    //amortizing JA/IA loads across the column dimension at the cost of higher register pressure.
    template <typename L, typename R, typename M, AssignmentMode Mode, int Batches,
        bool BroadcastMatrix = internal::traits<L>::BatchesAtCompileTime==1,
        bool BroadcastRhs = internal::traits<R>::BatchesAtCompileTime == 1>
    __global__ void __launch_bounds__(256) CSRMMKernel_StaticBatches(dim3 virtual_size, const L matrix, const R dense, M output, bool useOneThreadPerRow)
    {
        typedef typename L::Scalar LeftScalar;
        typedef typename R::Scalar RightScalar;
        typedef typename M::Scalar OutputScalar;
        typedef ProductElementFunctor<LeftScalar, RightScalar, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE> Functor;
        SparsityPattern<CSR>::IndexVector JA = matrix.getSparsityPattern().JA;
        SparsityPattern<CSR>::IndexVector IA = matrix.getSparsityPattern().IA;
        const int nnz = matrix.getSparsityPattern().nnz;
        if (useOneThreadPerRow)
        {
            //One thread per row, inner loops over all cols and batches.
            //Amortizes JA/IA loads across columns.
            const Index numCols = output.cols();
            CUMAT_KERNEL_1D_LOOP(row, virtual_size)
                int start = JA.getRawCoeff(row);
                int end = JA.getRawCoeff(row + 1);
                int len = end - start;
                OutputScalar value[CUMAT_SPARSE_MM_ACCUM_THRESHOLD] = {0};
                for (int t = 0; t < len; ++t)
                {
                    int inner = IA.getRawCoeff(start + t);
                    for (int c = 0; c < numCols; ++c)
                    {
                        int base = c * Batches;
#pragma unroll
                        for (int b = 0; b < Batches; ++b)
                        {
                            LeftScalar tmp1 = BroadcastMatrix
                                ? matrix.getSparseCoeff(row, inner, 0, start + t)
                                : matrix.getSparseCoeff(row, inner, b, start + t + b * nnz);
                            RightScalar tmp2 = dense.coeff(inner, c, BroadcastRhs ? 0 : b, -1);
                            value[base + b] += Functor::mult(tmp1, tmp2);
                        }
                    }
                }
                for (int c = 0; c < numCols; ++c)
                {
                    int base = c * Batches;
#pragma unroll
                    for (int b = 0; b < Batches; ++b)
                    {
                        Index linearIndex;
                        if (CUMAT_IS_COLUMN_MAJOR(traits<M>::Flags))
                            linearIndex = row + c * output.rows();
                        else
                            linearIndex = row * output.cols() + c;
                        linearIndex += b * output.rows() * output.cols();
                        internal::CwiseAssignmentHandler<M, OutputScalar, Mode>::assign(output, value[base + b], linearIndex);
                    }
                }
            CUMAT_KERNEL_1D_LOOP_END
        }
        else
        {
            //Default strategy: 2D loop with column-fast mapping, thread per (row, col) pair
            CUMAT_KERNEL_2D_LOOP(col, row, virtual_size)
                int start = JA.getRawCoeff(row);
                int end = JA.getRawCoeff(row + 1);
                int len = end - start;
                OutputScalar value[Batches];
#pragma unroll
                for (int b = 0; b < Batches; ++b) value[b] = OutputScalar(0);
                for (int t = 0; t < len; ++t)
                {
                    int inner = IA.getRawCoeff(start + t);
#pragma unroll
                    for (int b = 0; b < Batches; ++b)
                    {
                        LeftScalar tmp1 = BroadcastMatrix
                            ? matrix.getSparseCoeff(row, inner, 0, start + t)
                            : matrix.getSparseCoeff(row, inner, b, start + t + b * nnz);
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
    }

    //CSC SpMV kernel: one thread per column, uses shared memory hash table to
    //batch atomicAdd operations and reduce contention when multiple columns write
    //to the same output row.
    //Uses linear-probing hash table in shared memory: each entry stores (key, value).
    //After all columns are processed, the table is flushed to global memory via atomicAdd.
    //If the hash table fills up, falls back to direct global atomicAdd.
    enum { CSC_SMEM_SIZE = 1024 };
    template <typename L, typename R, typename M, AssignmentMode Mode, int Batches,
        bool BroadcastMatrix = internal::traits<L>::BatchesAtCompileTime==1,
        bool BroadcastRhs = internal::traits<R>::BatchesAtCompileTime == 1>
    __global__ void __launch_bounds__(256) CSCMVKernel_StaticBatches(dim3 virtual_size, const L matrix, const R vector, M output)
    {
        typedef typename L::Scalar LeftScalar;
        typedef typename R::Scalar RightScalar;
        typedef typename M::Scalar OutputScalar;
        typedef ProductElementFunctor<LeftScalar, RightScalar, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE> Functor;
        extern __shared__ char smem[];
        int* slot_key = (int*)smem;
        OutputScalar* slot_val = (OutputScalar*)(smem + CSC_SMEM_SIZE * sizeof(int));
        for (int i = threadIdx.x; i < CSC_SMEM_SIZE; i += blockDim.x) {
            slot_key[i] = -1;
            slot_val[i] = OutputScalar(0);
        }
        __syncthreads();
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
                    int key = inner + b * rows;
                    unsigned int slot = (unsigned int)(key * 2654435761U) % CSC_SMEM_SIZE;
                    int old = atomicCAS(&slot_key[slot], -1, key);
                    int probe = 0;
                    while (old != -1 && old != key) {
                        slot = (slot + 1) % CSC_SMEM_SIZE;
                        old = atomicCAS(&slot_key[slot], -1, key);
                        if (++probe >= CSC_SMEM_SIZE) {
                            atomicAdd(&outputData[key], tmp3);
                            goto next_batch;
                        }
                    }
                    atomicAdd(&slot_val[slot], tmp3);
                    next_batch:;
                }
            }
        CUMAT_KERNEL_1D_LOOP_END
        __syncthreads();
        for (int i = threadIdx.x; i < CSC_SMEM_SIZE; i += blockDim.x) {
            int key = slot_key[i];
            if (key != -1) {
                atomicAdd(&outputData[key], slot_val[i]);
            }
        }
    }

    //CSC SpMM kernel: sparse matrix * dense matrix -> dense matrix
    //Uses shared memory hash table to batch atomicAdd operations and reduce contention.
    template <typename L, typename R, typename M, AssignmentMode Mode, int Batches,
        bool BroadcastMatrix = internal::traits<L>::BatchesAtCompileTime==1,
        bool BroadcastRhs = internal::traits<R>::BatchesAtCompileTime == 1>
    __global__ void __launch_bounds__(256) CSCMMKernel_StaticBatches(dim3 virtual_size, const L matrix, const R dense, M output)
    {
        typedef typename L::Scalar LeftScalar;
        typedef typename R::Scalar RightScalar;
        typedef typename M::Scalar OutputScalar;
        typedef ProductElementFunctor<LeftScalar, RightScalar, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE> Functor;
        extern __shared__ char smem[];
        int* slot_key = (int*)smem;
        OutputScalar* slot_val = (OutputScalar*)(smem + CSC_SMEM_SIZE * sizeof(int));
        for (int i = threadIdx.x; i < CSC_SMEM_SIZE; i += blockDim.x) {
            slot_key[i] = -1;
            slot_val[i] = OutputScalar(0);
        }
        __syncthreads();
        SparsityPattern<CSC>::IndexVector JA = matrix.getSparsityPattern().JA;
        SparsityPattern<CSC>::IndexVector IA = matrix.getSparsityPattern().IA;
        const int nnz = matrix.getSparsityPattern().nnz;
        const int rows = matrix.rows();
        const int cols = output.cols();
        const int rowStride = (CUMAT_IS_COLUMN_MAJOR(traits<M>::Flags)) ? 1 : output.cols();
        const int colStride = (CUMAT_IS_COLUMN_MAJOR(traits<M>::Flags)) ? output.rows() : 1;
        const int batchStride = output.rows() * output.cols();
        OutputScalar* outputData = output.data();
        CUMAT_KERNEL_1D_LOOP(outer, virtual_size)
            int start = JA.getRawCoeff(outer);
            int end = JA.getRawCoeff(outer + 1);
            if (start >= end) continue;
            for (int i = start; i < end; ++i)
            {
                int inner = IA.getRawCoeff(i);
                for (int c = 0; c < cols; ++c)
                {
#pragma unroll
                    for (int b = 0; b < Batches; ++b) {
                        LeftScalar tmp1 = BroadcastMatrix
                            ? matrix.getSparseCoeff(inner, outer, 0, i)
                            : matrix.getSparseCoeff(inner, outer, b, i + b * nnz);
                        RightScalar tmp2 = dense.coeff(outer, c, BroadcastRhs ? 0 : b, -1);
                        OutputScalar tmp3 = Functor::mult(tmp1, tmp2);
                        int key = inner * rowStride + c * colStride + b * batchStride;
                        unsigned int slot = (unsigned int)(key * 2654435761U) % CSC_SMEM_SIZE;
                        int old = atomicCAS(&slot_key[slot], -1, key);
                        int probe = 0;
                        while (old != -1 && old != key) {
                            slot = (slot + 1) % CSC_SMEM_SIZE;
                            old = atomicCAS(&slot_key[slot], -1, key);
                            if (++probe >= CSC_SMEM_SIZE) {
                                atomicAdd(&outputData[key], tmp3);
                                goto next_batch_mm;
                            }
                        }
                        atomicAdd(&slot_val[slot], tmp3);
                        next_batch_mm:;
                    }
                }
            }
        CUMAT_KERNEL_1D_LOOP_END
        __syncthreads();
        for (int i = threadIdx.x; i < CSC_SMEM_SIZE; i += blockDim.x) {
            int key = slot_key[i];
            if (key != -1) {
                atomicAdd(&outputData[key], slot_val[i]);
            }
        }
    }

    }

	namespace kernels
	{
		//ELLPACK Matrix-Vector kernel. One thread per row
		template <typename L, typename R, typename M, AssignmentMode Mode, int Batches,
			bool BroadcastMatrix = internal::traits<L>::BatchesAtCompileTime == 1,
			bool BroadcastRhs = internal::traits<R>::BatchesAtCompileTime == 1>
			__global__ void __launch_bounds__(256) ELLPACKMVKernel_StaticBatches(dim3 virtual_size, const L matrix, const R vector, M output)
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
					if (col < 0) continue;
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

		//ELLPACK SpMM kernel: sparse matrix * dense matrix -> dense matrix
		//One thread per (row, col) of the output
		//When useOneThreadPerRow is true, one thread per row iterates over all cols and batches,
		//amortizing index loads across the column dimension.
		template <typename L, typename R, typename M, AssignmentMode Mode, int Batches,
			bool BroadcastMatrix = internal::traits<L>::BatchesAtCompileTime == 1,
			bool BroadcastRhs = internal::traits<R>::BatchesAtCompileTime == 1>
		__global__ void __launch_bounds__(256) ELLPACKMMKernel_StaticBatches(dim3 virtual_size, const L matrix, const R dense, M output, bool useOneThreadPerRow)
		{
			typedef typename L::Scalar LeftScalar;
			typedef typename R::Scalar RightScalar;
			typedef typename M::Scalar OutputScalar;
			typedef ProductElementFunctor<LeftScalar, RightScalar, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE> Functor;
			const SparsityPattern<SparseFlags::ELLPACK>::IndexMatrix& indices = matrix.getSparsityPattern().indices;
			const int nnzPerRow = matrix.getSparsityPattern().nnzPerRow;
			const int rows = matrix.getSparsityPattern().rows;
			if (useOneThreadPerRow)
			{
				//One thread per row, inner loops over all cols and batches.
				//Amortizes index loads across columns.
				const Index numCols = output.cols();
				CUMAT_KERNEL_1D_LOOP(row, virtual_size)
					OutputScalar value[CUMAT_SPARSE_MM_ACCUM_THRESHOLD] = {0};
					for (int ci = 0; ci < nnzPerRow; ++ci)
					{
						int col = indices.coeff(row, ci, 0, -1);
						if (col < 0) continue;
						for (int c = 0; c < numCols; ++c)
						{
							int base = c * Batches;
#pragma unroll
							for (int b = 0; b < Batches; ++b)
							{
								LeftScalar tmp1 = BroadcastMatrix
									? matrix.getSparseCoeff(row, col, 0, row + ci*rows)
									: matrix.getSparseCoeff(row, col, b, row + rows*(ci + b*nnzPerRow));
								RightScalar tmp2 = dense.coeff(col, c, BroadcastRhs ? 0 : b, -1);
								value[base + b] += Functor::mult(tmp1, tmp2);
							}
						}
					}
					for (int c = 0; c < numCols; ++c)
					{
						int base = c * Batches;
#pragma unroll
						for (int b = 0; b < Batches; ++b)
						{
							Index linearIndex;
							if (CUMAT_IS_COLUMN_MAJOR(traits<M>::Flags))
								linearIndex = row + c * output.rows();
							else
								linearIndex = row * output.cols() + c;
							linearIndex += b * output.rows() * output.cols();
							internal::CwiseAssignmentHandler<M, OutputScalar, Mode>::assign(output, value[base + b], linearIndex);
						}
					}
				CUMAT_KERNEL_1D_LOOP_END
			}
			else
			{
				//Default strategy: 2D loop, thread per (row, colOut) pair
				CUMAT_KERNEL_2D_LOOP(row, colOut, virtual_size)
					OutputScalar value[Batches] = {0};
					for (int ci = 0; ci < nnzPerRow; ++ci) {
						int col = indices.coeff(row, ci, 0, -1);
						if (col < 0) continue;
#pragma unroll
						for (int b = 0; b < Batches; ++b) {
							LeftScalar tmp1 = BroadcastMatrix
								? matrix.getSparseCoeff(row, col, 0, row + ci*rows)
								: matrix.getSparseCoeff(row, col, b, row + rows*(ci + b*nnzPerRow));
							RightScalar tmp2 = dense.coeff(col, colOut, BroadcastRhs ? 0 : b, -1);
							OutputScalar tmp3 = Functor::mult(tmp1, tmp2);
							value[b] += tmp3;
						}
					}
#pragma unroll
					for (int b = 0; b < Batches; ++b) {
						Index linearIndex;
						if (CUMAT_IS_COLUMN_MAJOR(traits<M>::Flags))
							linearIndex = row + colOut * output.rows();
						else
							linearIndex = row * output.cols() + colOut;
						linearIndex += b * output.rows() * output.cols();
						internal::CwiseAssignmentHandler<M, OutputScalar, Mode>::assign(output, value[b], linearIndex);
					}
				CUMAT_KERNEL_2D_LOOP_END
			}
		}

	}

    //Merged SparseMatrix (CSR/CSC/ELLPACK) * dense -> dense product
    //Dispatches to SpMV (vector RHS) or SpMM (matrix RHS)
    template<
        typename _Dst,
        typename _SrcLeftScalar, int _SrcLeftBatches, int _SFlag,
        typename _SrcRight,
        AssignmentMode _AssignmentMode
    >
    struct ProductAssignment<
        _Dst, DenseDstTag, ProductArgOp::NONE, 
        SparseMatrix<_SrcLeftScalar, _SrcLeftBatches, _SFlag>, CwiseSrcTag, ProductArgOp::NONE,
        _SrcRight, CwiseSrcTag, ProductArgOp::NONE, 
        _AssignmentMode>
    {
        using SrcLeft = SparseMatrix<_SrcLeftScalar, _SrcLeftBatches, _SFlag>;
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
            if (_SFlag == CSC) dst.setZero();
            if (Op::ColumnsRight == 1)
            {
                CUMAT_LOG_DEBUG("Evaluate " << formatName() << " SparseMatrix-DenseVector multiplication " << internal::type_name<decltype(op.derived())>()
                    << " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols());
                Index mvSize = (_SFlag == CSC) ? op.left().cols() : dst.rows();
                spMVLaunch<DstActual>(ctx, op, dst, mvSize);
            } else {
                CUMAT_LOG_DEBUG("Evaluate " << formatName() << " SparseMatrix-DenseMatrix multiplication " << internal::type_name<decltype(op.derived())>()
                    << " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols()
                    << ", rhsCols=" << op.right().cols());
                dim3 virtualSize(dst.rows(), dst.cols(), 1);
                spMMLaunch<DstActual>(ctx, op, dst, virtualSize);
            }
            CUMAT_CHECK_ERROR();
            CUMAT_LOG_DEBUG("Evaluation done");
        }

    private:
        static const char* formatName() {
            if (_SFlag == CSR) return "CSR";
            if (_SFlag == CSC) return "CSC";
            return "ELLPACK";
        }

        template<typename DstActual>
        static void spMVLaunch(Context& ctx, const Op& op, _Dst& dst, Index mvSize) {
            spMVLaunchImpl<DstActual>(ctx, op, dst, mvSize, std::integral_constant<int, _SFlag>());
        }
        template<typename DstActual>
        static void spMVLaunchImpl(Context& ctx, const Op& op, _Dst& dst, Index mvSize, std::integral_constant<int, CSR>) {
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(mvSize, kernels::CSRMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
            kernels::CSRMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
                (cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
        }
        template<typename DstActual>
        static void spMVLaunchImpl(Context& ctx, const Op& op, _Dst& dst, Index mvSize, std::integral_constant<int, CSC>) {
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(mvSize, kernels::CSCMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
            size_t smemSize = kernels::CSC_SMEM_SIZE * (sizeof(int) + sizeof(typename DstActual::Scalar));
            kernels::CSCMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                <<<cfg.block_count, cfg.thread_per_block, smemSize, ctx.stream() >>>
                (cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
        }
        template<typename DstActual>
        static void spMVLaunchImpl(Context& ctx, const Op& op, _Dst& dst, Index mvSize, std::integral_constant<int, ELLPACK>) {
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(mvSize, kernels::ELLPACKMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
            kernels::ELLPACKMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
                (cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
        }

        template<typename DstActual>
        static void spMMLaunch(Context& ctx, const Op& op, _Dst& dst, dim3 virtualSize) {
            spMMLaunchImpl<DstActual>(ctx, op, dst, virtualSize, std::integral_constant<int, _SFlag>());
        }
        template<typename DstActual>
        static void spMMLaunchImpl(Context& ctx, const Op& op, _Dst& dst, dim3 virtualSize, std::integral_constant<int, CSR>) {
            const Index cols = dst.cols();
            const bool useOneThreadPerRow = cols * Op::Batches <= CUMAT_SPARSE_MM_ACCUM_THRESHOLD;
            if (useOneThreadPerRow)
            {
                KernelLaunchConfig cfg = ctx.createLaunchConfig1D(virtualSize.x, kernels::CSRMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
                cfg.virtual_size.y = static_cast<unsigned int>(cols);
                kernels::CSRMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                    <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
                    (cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived(), true);
            }
            else
            {
                KernelLaunchConfig cfg = ctx.createLaunchConfig1D(virtualSize.x * virtualSize.y, kernels::CSRMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
                // Swap virtualSize to column-fast: (cols, rows) so adjacent threads share the same row
                dim3 vsColFast(virtualSize.y, virtualSize.x);
                kernels::CSRMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                    <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
                    (vsColFast, op.derived().left().derived(), op.derived().right().derived(), dst.derived(), false);
            }
        }
        template<typename DstActual>
        static void spMMLaunchImpl(Context& ctx, const Op& op, _Dst& dst, dim3 virtualSize, std::integral_constant<int, CSC>) {
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(virtualSize.x * virtualSize.y, kernels::CSCMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
            size_t smemSize = kernels::CSC_SMEM_SIZE * (sizeof(int) + sizeof(typename DstActual::Scalar));
            kernels::CSCMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                <<<cfg.block_count, cfg.thread_per_block, smemSize, ctx.stream() >>>
                (virtualSize, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
        }
        template<typename DstActual>
        static void spMMLaunchImpl(Context& ctx, const Op& op, _Dst& dst, dim3 virtualSize, std::integral_constant<int, ELLPACK>) {
            const Index cols = dst.cols();
            const bool useOneThreadPerRow = cols * Op::Batches <= CUMAT_SPARSE_MM_ACCUM_THRESHOLD;
            if (useOneThreadPerRow)
            {
                KernelLaunchConfig cfg = ctx.createLaunchConfig1D(virtualSize.x, kernels::ELLPACKMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
                cfg.virtual_size.y = static_cast<unsigned int>(cols);
                kernels::ELLPACKMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                    <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
                    (cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived(), true);
            }
            else
            {
                KernelLaunchConfig cfg = ctx.createLaunchConfig1D(virtualSize.x * virtualSize.y, kernels::ELLPACKMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
                kernels::ELLPACKMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                    <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
                    (virtualSize, op.derived().left().derived(), op.derived().right().derived(), dst.derived(), false);
            }
        }
    };

    //CwiseSrcTag (Sparse) * CwiseSrcTag (Dense) -> DenseDstTag, CSC sparse matrix-dense product
    //Dispatches to SpMV (vector RHS) or SpMM (matrix RHS)
    //CSC SparseMatrix
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

        CUMAT_STATIC_ASSERT((Op::Batches != Dynamic),
                "CSC SparseMatrix - dense product does only support compile-time fixed batch count");

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
            dst.setZero();
            if (Op::ColumnsRight == 1)
            {
                //SpMV: matrix * vector
                CUMAT_LOG_DEBUG("Evaluate CSC SparseMatrix-DenseVector multiplication " << internal::type_name<decltype(op.derived())>()
                    << " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols());
                size_t smemSz = kernels::CSC_SMEM_SIZE * (sizeof(int) + sizeof(typename DstActual::Scalar));
                KernelLaunchConfig cfg = ctx.createLaunchConfig1D(op.left().cols(), kernels::CSCMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
    			kernels::CSCMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                    <<<cfg.block_count, cfg.thread_per_block, smemSz, ctx.stream()>>>
                    (cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
            } else {
                //SpMM: matrix * dense matrix
                CUMAT_LOG_DEBUG("Evaluate CSC SparseMatrix-DenseMatrix multiplication " << internal::type_name<decltype(op.derived())>()
                    << " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols()
                    << ", rhsCols=" << op.right().cols());
                size_t smemSz = kernels::CSC_SMEM_SIZE * (sizeof(int) + sizeof(typename DstActual::Scalar));
                dim3 virtualSize(dst.rows(), dst.cols(), 1);
                KernelLaunchConfig cfg = ctx.createLaunchConfig1D(virtualSize.x * virtualSize.y, kernels::CSCMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
    			kernels::CSCMMKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                    <<<cfg.block_count, cfg.thread_per_block, smemSz, ctx.stream() >>>
                    (virtualSize, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
            }
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

			CUMAT_LOG_DEBUG("Evaluate SparseMatrix-DenseVector multiplication " << internal::type_name<decltype(op.derived())>()
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

    //CwiseSrcTag (SparseExpressionOp) * CwiseSrcTag (Dense-Vector) -> DenseDstTag (Vector-Vector), sparse matrix-vector product
    //ELLPACK SparseExpressionOp
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

            CUMAT_LOG_DEBUG("Evaluate ELLPACK SparseExpressionOp-DenseVector multiplication " << internal::type_name<decltype(op.derived())>()
                << " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols());;

            Context& ctx = Context::current();
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(static_cast<unsigned int>(dst.size()), kernels::ELLPACKMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
            kernels::ELLPACKMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
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

            CUMAT_LOG_DEBUG("Evaluate CSC SparseExpressionOp-DenseVector multiplication " << internal::type_name<decltype(op.derived())>()
                << " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols());;

            Context& ctx = Context::current();
            dst.setZero();
            size_t smemSz = kernels::CSC_SMEM_SIZE * (sizeof(int) + sizeof(typename DstActual::Scalar));
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(op.left().cols(), kernels::CSCMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>);
            kernels::CSCMVKernel_StaticBatches<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode, Op::Batches>
                <<<cfg.block_count, cfg.thread_per_block, smemSz, ctx.stream() >>>
                (cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
            CUMAT_CHECK_ERROR();
            CUMAT_LOG_DEBUG("Evaluation done");
        }
    };
#endif
}

CUMAT_NAMESPACE_END

#endif