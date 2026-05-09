#ifndef __CUMAT_SPARSE_EVALUATION__
#define __CUMAT_SPARSE_EVALUATION__

#include "Macros.h"
#include "CwiseOp.h"
#include "SparseMatrix.h"

#ifndef CUMAT_SPARSE_EVAL_BATCH_THRESHOLD
/**
 * \brief Threshold for batch-inner strategy in sparse cwise evaluation kernels.
 * When batches exceeds this value, the kernel switches from a 2D (outer, batch)
 * parallel strategy to a batch-inner strategy where one thread per outer index
 * iterates over all batches. This amortizes JA/IA sparsity-pattern loads across
 * batches and improves performance when batches >> 1.
 * Set to 0 to always use batch-inner, or to INT_MAX to always use 2D.
 * Default: 4
 */
#define CUMAT_SPARSE_EVAL_BATCH_THRESHOLD 4
#endif

CUMAT_NAMESPACE_BEGIN

namespace internal
{
	namespace kernels
	{
		template <typename T, typename M, AssignmentMode Mode>
		__global__ void __launch_bounds__(256) CwiseCSREvaluationKernel(dim3 virtual_size, const T expr, M matrix, bool useBatchInner)
		{
			const int* JA = matrix.getSparsityPattern().JA.data();
			const int* IA = matrix.getSparsityPattern().IA.data();
			Index batchStride = matrix.getSparsityPattern().nnz;
			if (useBatchInner)
			{
				//Batch-inner strategy: one thread per outer index, iterates all batches
				//JA/IA loaded once per thread, reused across batches
				CUMAT_KERNEL_1D_LOOP(outer, virtual_size)
					int start = JA[outer];
					int end = JA[outer + 1];
					Index numBatches = virtual_size.y;
					for (Index batch = 0; batch < numBatches; ++batch)
						for (int i = start; i < end; ++i)
						{
							int inner = IA[i];
							Index idx = i + batch * batchStride;
							auto val = expr.coeff(outer, inner, batch, idx);
							internal::CwiseAssignmentHandler<M, decltype(val), Mode>::assign(matrix, val, idx);
						}
				CUMAT_KERNEL_1D_LOOP_END
			}
			else
			{
				//Default strategy: 2D loop, thread per (outer, batch) pair
				CUMAT_KERNEL_2D_LOOP(outer, batch, virtual_size)
					int start = JA[outer];
					int end = JA[outer + 1];
					for (int i = start; i < end; ++i)
					{
						int inner = IA[i];
						Index idx = i + batch * batchStride;
						auto val = expr.coeff(outer, inner, batch, idx);
						internal::CwiseAssignmentHandler<M, decltype(val), Mode>::assign(matrix, val, idx);
					}
				CUMAT_KERNEL_2D_LOOP_END
			}
		}
		template <typename T, typename M, AssignmentMode Mode>
		__global__ void __launch_bounds__(256) CwiseCSCEvaluationKernel(dim3 virtual_size, const T expr, M matrix, bool useBatchInner)
		{
			const int* JA = matrix.getSparsityPattern().JA.data();
			const int* IA = matrix.getSparsityPattern().IA.data();
			Index batchStride = matrix.getSparsityPattern().nnz;
			if (useBatchInner)
			{
				//Batch-inner strategy: one thread per outer index, iterates all batches
				CUMAT_KERNEL_1D_LOOP(outer, virtual_size)
					int start = JA[outer];
					int end = JA[outer + 1];
					Index numBatches = virtual_size.y;
					for (Index batch = 0; batch < numBatches; ++batch)
						for (int i = start; i < end; ++i)
						{
							int inner = IA[i];
							Index idx = i + batch * batchStride;
							auto val = expr.coeff(inner, outer, batch, idx);
							internal::CwiseAssignmentHandler<M, decltype(val), Mode>::assign(matrix, val, idx);
						}
				CUMAT_KERNEL_1D_LOOP_END
			}
			else
			{
				//Default strategy: 2D loop, thread per (outer, batch) pair
				CUMAT_KERNEL_2D_LOOP(outer, batch, virtual_size)
					int start = JA[outer];
					int end = JA[outer + 1];
					for (int i = start; i < end; ++i)
					{
						int inner = IA[i];
						Index idx = i + batch * batchStride;
						auto val = expr.coeff(inner, outer, batch, idx);
						internal::CwiseAssignmentHandler<M, decltype(val), Mode>::assign(matrix, val, idx);
					}
				CUMAT_KERNEL_2D_LOOP_END
			}
		}
		template <typename T, typename M, AssignmentMode Mode>
		__global__ void __launch_bounds__(256) CwiseELLPACKEvaluationKernel(dim3 virtual_size, const T expr, M matrix, bool useBatchInner)
		{
			const SparsityPattern<SparseFlags::ELLPACK>::IndexMatrix indices = matrix.getSparsityPattern().indices;
			Index nnzPerRow = matrix.getSparsityPattern().nnzPerRow;
			Index rows = matrix.getSparsityPattern().rows;
			Index batchStride = rows * nnzPerRow;
			if (useBatchInner)
			{
				//Batch-inner strategy: one thread per row, iterates all batches
				CUMAT_KERNEL_1D_LOOP(row, virtual_size)
					Index numBatches = virtual_size.y;
					for (Index batch = 0; batch < numBatches; ++batch)
						for (int i = 0; i < nnzPerRow; ++i)
						{
							Index col = indices.coeff(row, i, 0, -1);
							if (col >= 0)
							{
								Index idx = row + i * rows + batch * batchStride;
								auto val = expr.coeff(row, col, batch, idx);
								internal::CwiseAssignmentHandler<M, decltype(val), Mode>::assign(matrix, val, idx);
							}
						}
				CUMAT_KERNEL_1D_LOOP_END
			}
			else
			{
				//Default strategy: 2D loop, thread per (row, batch) pair
				CUMAT_KERNEL_2D_LOOP(row, batch, virtual_size)
					for (int i = 0; i < nnzPerRow; ++i)
					{
						Index col = indices.coeff(row, i, 0, -1);
						if (col >= 0)
						{
							Index idx = row + i * rows + batch * batchStride;
							auto val = expr.coeff(row, col, batch, idx);
							internal::CwiseAssignmentHandler<M, decltype(val), Mode>::assign(matrix, val, idx);
						}
					}
				CUMAT_KERNEL_2D_LOOP_END
			}
		}
	}
}

namespace internal {

#if CUMAT_NVCC==1
    //General assignment for everything that fulfills CwiseSrcTag into SparseDstTag (cwise sparse evaluation)
    //The source expression is only evaluated at the non-zero entries of the target SparseMatrix
    template<typename _Dst, typename _Src, AssignmentMode _Mode>
    struct Assignment<_Dst, _Src, _Mode, SparseDstTag, CwiseSrcTag>
    {
    private:
		static void assign(_Dst& dst, const _Src& src, std::integral_constant<int, SparseFlags::CSR>)
		{
			Context& ctx = Context::current();
			typedef typename _Src::Type SrcType;
			typedef typename _Dst::Type DstType;
			const Index batches = dst.derived().batches();
			const bool useBatchInner = batches > CUMAT_SPARSE_EVAL_BATCH_THRESHOLD;
			KernelLaunchConfig cfg;
			if (useBatchInner)
			{
				cfg = ctx.createLaunchConfig1D(dst.derived().outerSize(),
					kernels::CwiseCSREvaluationKernel<SrcType, DstType, _Mode>);
				cfg.virtual_size.y = static_cast<unsigned int>(batches);
			}
			else
			{
				cfg = ctx.createLaunchConfig2D(
					static_cast<unsigned int>(dst.derived().outerSize()),
					static_cast<unsigned int>(batches),
					kernels::CwiseCSREvaluationKernel<SrcType, DstType, _Mode>);
			}
			kernels::CwiseCSREvaluationKernel<SrcType, DstType, _Mode>
				<<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream()>>>(cfg.virtual_size, src.derived(), dst.derived(), useBatchInner);
			CUMAT_CHECK_ERROR();
		}
		static void assign(_Dst& dst, const _Src& src, std::integral_constant<int, SparseFlags::CSC>)
		{
			Context& ctx = Context::current();
			typedef typename _Src::Type SrcType;
			typedef typename _Dst::Type DstType;
			const Index batches = dst.derived().batches();
			const bool useBatchInner = batches > CUMAT_SPARSE_EVAL_BATCH_THRESHOLD;
			KernelLaunchConfig cfg;
			if (useBatchInner)
			{
				cfg = ctx.createLaunchConfig1D(dst.derived().outerSize(),
					kernels::CwiseCSCEvaluationKernel<SrcType, DstType, _Mode>);
				cfg.virtual_size.y = static_cast<unsigned int>(batches);
			}
			else
			{
				cfg = ctx.createLaunchConfig2D(
					static_cast<unsigned int>(dst.derived().outerSize()),
					static_cast<unsigned int>(batches),
					kernels::CwiseCSCEvaluationKernel<SrcType, DstType, _Mode>);
			}
			kernels::CwiseCSCEvaluationKernel<SrcType, DstType, _Mode>
				<<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream()>>>(cfg.virtual_size, src.derived(), dst.derived(), useBatchInner);
			CUMAT_CHECK_ERROR();
		}
		static void assign(_Dst& dst, const _Src& src, std::integral_constant<int, SparseFlags::ELLPACK>)
		{
			Context& ctx = Context::current();
			typedef typename _Src::Type SrcType;
			typedef typename _Dst::Type DstType;
			const Index batches = dst.derived().batches();
			const bool useBatchInner = batches > CUMAT_SPARSE_EVAL_BATCH_THRESHOLD;
			KernelLaunchConfig cfg;
			if (useBatchInner)
			{
				cfg = ctx.createLaunchConfig1D(dst.derived().outerSize(),
					kernels::CwiseELLPACKEvaluationKernel<SrcType, DstType, _Mode>);
				cfg.virtual_size.y = static_cast<unsigned int>(batches);
			}
			else
			{
				cfg = ctx.createLaunchConfig2D(
					static_cast<unsigned int>(dst.derived().outerSize()),
					static_cast<unsigned int>(batches),
					kernels::CwiseELLPACKEvaluationKernel<SrcType, DstType, _Mode>);
			}
			kernels::CwiseELLPACKEvaluationKernel<SrcType, DstType, _Mode>
				<<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream()>>>(cfg.virtual_size, src.derived(), dst.derived(), useBatchInner);
			CUMAT_CHECK_ERROR();
		}

    public:
        static void assign(_Dst& dst, const _Src& src)
        {
            typedef typename _Dst::Type DstActual;
            typedef typename _Src::Type SrcActual;
            CUMAT_PROFILING_INC(EvalCwiseSparse);
            CUMAT_PROFILING_INC(EvalAny);
            if (dst.size() == 0) return;
            CUMAT_ASSERT(src.rows() == dst.rows());
            CUMAT_ASSERT(src.cols() == dst.cols());
            CUMAT_ASSERT(src.batches() == dst.batches());

            CUMAT_LOG_DEBUG("Evaluate component wise sparse expression " << internal::type_name<decltype(src.derived())>()
				<< "\n rows=" << src.rows() << ", cols=" << src.cols() << ", batches=" << src.batches());
			assign(dst, src, std::integral_constant<int, DstActual::SFlags>());
            CUMAT_LOG_DEBUG("Evaluation done");
        }
    };
#endif
}

CUMAT_NAMESPACE_END

#endif