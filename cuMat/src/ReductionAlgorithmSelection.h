#ifndef __CUMAT_REDUCTION_ALGORITHM_SELECTION_H__
#define __CUMAT_REDUCTION_ALGORITHM_SELECTION_H__

#include <tuple>
#include <array>

#include "Macros.h"
#include "ForwardDeclarations.h"

CUMAT_NAMESPACE_BEGIN

namespace internal
{
	/**
	 * \brief Algorithms possible during dynamic selection.
	 * These are a subset of the tags from namespace ReductionAlg.
	 */
	enum class ReductionAlgorithm
	{
		Segmented,
		Thread,
		Warp,
		Block256,
		Device1,
		Device2,
		Device4
	};

	/**
	 * \brief Selects the best reduction algorithm dynamically given
	 * the reduction axis (inner, middle, outer).
	 * Given a matrix in ColumnMajor order, these axis
	 * correspond to the following: inner=Row, middle=Column, outer=Batch.
	 * 
	 * Each decision boundary is a linear inequality in log2 space:
	 *   a * log2(numBatches) + b * log2(batchSize) < c
	 * 
	 * Default thresholds were tuned on an Nvidia RTX 2070.
	 * Override any CUMAT_REDUCTION_* macro at compile time to adapt
	 * to a different GPU architecture.
	 */
	struct ReductionAlgorithmSelection
	{
	private:
		typedef std::tuple<double, double, double> condition;
		static constexpr int MAX_CONDITION = 3;
		typedef std::array<condition, MAX_CONDITION> conditions;
		typedef std::tuple<ReductionAlgorithm, int, conditions> choice;

		template<int N>
		static ReductionAlgorithm select(
			const choice(&conditions)[N], ReductionAlgorithm def,
			Index numBatches, Index batchSize)
		{
			const double nb = std::log2(numBatches);
			const double bs = std::log2(batchSize);
			for (const choice& c : conditions)
			{
				bool success = true;
				for (int i = 0; i < std::get<1>(c); ++i) {
					const auto& cond = std::get<2>(c)[i];
					if (std::get<0>(cond)*nb + std::get<1>(cond)*bs < std::get<2>(cond))
						success = false;
				}
				if (success)
					return std::get<0>(c);
			}
			return def;
		}

	public:
		//=========================================================================
		// Tuning knobs — override any at compile time via -D flag
		// Each threshold (c) is the RHS of:
		//   a * log2(numBatches) + b * log2(batchSize) >= c
		// to enter the corresponding algorithm's region.
		//=========================================================================

		//--- Inner axis (row reduction) ---
#ifndef CUMAT_REDUCTION_INNER_DEVICE1_THRESH
#define CUMAT_REDUCTION_INNER_DEVICE1_THRESH 19.5
#endif
#ifndef CUMAT_REDUCTION_INNER_DEVICE1_NB_CAP
#define CUMAT_REDUCTION_INNER_DEVICE1_NB_CAP 2.5
#endif
#ifndef CUMAT_REDUCTION_INNER_DEVICE2_THRESH
#define CUMAT_REDUCTION_INNER_DEVICE2_THRESH 17.821428571428573
#endif
#ifndef CUMAT_REDUCTION_INNER_DEVICE2_NB_MIN
#define CUMAT_REDUCTION_INNER_DEVICE2_NB_MIN 2.5
#endif
#ifndef CUMAT_REDUCTION_INNER_DEVICE2_NB_MAX
#define CUMAT_REDUCTION_INNER_DEVICE2_NB_MAX 4.25
#endif
#ifndef CUMAT_REDUCTION_INNER_DEVICE4_THRESH
#define CUMAT_REDUCTION_INNER_DEVICE4_THRESH 16.25
#endif
#ifndef CUMAT_REDUCTION_INNER_DEVICE4_NB_MIN
#define CUMAT_REDUCTION_INNER_DEVICE4_NB_MIN 4.25
#endif
#ifndef CUMAT_REDUCTION_INNER_DEVICE4_NB_MAX
#define CUMAT_REDUCTION_INNER_DEVICE4_NB_MAX 5.5
#endif
#ifndef CUMAT_REDUCTION_INNER_BLOCK256_THRESH
#define CUMAT_REDUCTION_INNER_BLOCK256_THRESH 8.0
#endif
#ifndef CUMAT_REDUCTION_INNER_BLOCK256_NB_MAX
#define CUMAT_REDUCTION_INNER_BLOCK256_NB_MAX 5.0
#endif
#ifndef CUMAT_REDUCTION_INNER_THREAD_THRESH
#define CUMAT_REDUCTION_INNER_THREAD_THRESH 2.01875
#endif
#ifndef CUMAT_REDUCTION_INNER_THREAD_BS_MIN
#define CUMAT_REDUCTION_INNER_THREAD_BS_MIN 4.75
#endif

		static ReductionAlgorithm inner(Index numBatches, Index batchSize)
		{
			static const choice CONDITIONS[] = {
				choice{ReductionAlgorithm::Device1, 2, {
					condition{1.2, 1.0, CUMAT_REDUCTION_INNER_DEVICE1_THRESH},
					condition{-1, 0, -CUMAT_REDUCTION_INNER_DEVICE1_NB_CAP}
				}},
				choice{ReductionAlgorithm::Device2, 3, {
					condition{0.42857142857142855, 1, CUMAT_REDUCTION_INNER_DEVICE2_THRESH},
					condition{-1, 0, -CUMAT_REDUCTION_INNER_DEVICE2_NB_MAX},
					condition{1, 0, CUMAT_REDUCTION_INNER_DEVICE2_NB_MIN}
				}},
				choice{ReductionAlgorithm::Device4, 3, {
					condition{0, 1, CUMAT_REDUCTION_INNER_DEVICE4_THRESH},
					condition{-1, 0, -CUMAT_REDUCTION_INNER_DEVICE4_NB_MAX},
					condition{1, 0, CUMAT_REDUCTION_INNER_DEVICE4_NB_MIN}
				}},
				choice{ReductionAlgorithm::Block256, 2, {
					condition{-1.6, 1, CUMAT_REDUCTION_INNER_BLOCK256_THRESH},
					condition{-1, 0, -CUMAT_REDUCTION_INNER_BLOCK256_NB_MAX}
				}},
				choice{ReductionAlgorithm::Thread, 2, {
					condition{0.475, -1, CUMAT_REDUCTION_INNER_THREAD_THRESH},
					condition{0, -1, -CUMAT_REDUCTION_INNER_THREAD_BS_MIN}
				}}
			};
			static const ReductionAlgorithm DEFAULT = ReductionAlgorithm::Warp;
			return select(CONDITIONS, DEFAULT, numBatches, batchSize);
		}

		//--- Middle axis (column reduction) ---
#ifndef CUMAT_REDUCTION_MIDDLE_DEVICE1_THRESH
#define CUMAT_REDUCTION_MIDDLE_DEVICE1_THRESH 19.5
#endif
#ifndef CUMAT_REDUCTION_MIDDLE_DEVICE1_NB_CAP
#define CUMAT_REDUCTION_MIDDLE_DEVICE1_NB_CAP 2.5
#endif
#ifndef CUMAT_REDUCTION_MIDDLE_DEVICE2_THRESH
#define CUMAT_REDUCTION_MIDDLE_DEVICE2_THRESH 15.5
#endif
#ifndef CUMAT_REDUCTION_MIDDLE_DEVICE2_NB_MIN
#define CUMAT_REDUCTION_MIDDLE_DEVICE2_NB_MIN 2.5
#endif
#ifndef CUMAT_REDUCTION_MIDDLE_DEVICE2_NB_MAX
#define CUMAT_REDUCTION_MIDDLE_DEVICE2_NB_MAX 4.0
#endif
#ifndef CUMAT_REDUCTION_MIDDLE_DEVICE4_THRESH
#define CUMAT_REDUCTION_MIDDLE_DEVICE4_THRESH 15.75
#endif
#ifndef CUMAT_REDUCTION_MIDDLE_DEVICE4_NB_MIN
#define CUMAT_REDUCTION_MIDDLE_DEVICE4_NB_MIN 4.0
#endif
#ifndef CUMAT_REDUCTION_MIDDLE_DEVICE4_NB_MAX
#define CUMAT_REDUCTION_MIDDLE_DEVICE4_NB_MAX 5.75
#endif
#ifndef CUMAT_REDUCTION_MIDDLE_BLOCK256_THRESH
#define CUMAT_REDUCTION_MIDDLE_BLOCK256_THRESH 9.0
#endif
#ifndef CUMAT_REDUCTION_MIDDLE_BLOCK256_NB_MAX
#define CUMAT_REDUCTION_MIDDLE_BLOCK256_NB_MAX 2.5
#endif
#ifndef CUMAT_REDUCTION_MIDDLE_WARP_THRESH
#define CUMAT_REDUCTION_MIDDLE_WARP_THRESH 4.0
#endif
#ifndef CUMAT_REDUCTION_MIDDLE_WARP_NB_MAX
#define CUMAT_REDUCTION_MIDDLE_WARP_NB_MAX 11.75
#endif

		static ReductionAlgorithm middle(Index numBatches, Index batchSize)
		{
			static const choice CONDITIONS[] = {
				choice{ReductionAlgorithm::Device1, 2, {
					condition{1.5, 1, CUMAT_REDUCTION_MIDDLE_DEVICE1_THRESH},
					condition{-1, 0, -CUMAT_REDUCTION_MIDDLE_DEVICE1_NB_CAP}
				}},
				choice{ReductionAlgorithm::Device2, 3, {
					condition{0, 1, CUMAT_REDUCTION_MIDDLE_DEVICE2_THRESH},
					condition{1, 0, CUMAT_REDUCTION_MIDDLE_DEVICE2_NB_MIN},
					condition{-1, 0, -CUMAT_REDUCTION_MIDDLE_DEVICE2_NB_MAX}
				}},
				choice{ReductionAlgorithm::Device4, 3, {
					condition{0, 1, CUMAT_REDUCTION_MIDDLE_DEVICE4_THRESH},
					condition{1, 0, CUMAT_REDUCTION_MIDDLE_DEVICE4_NB_MIN},
					condition{-1, 0, -CUMAT_REDUCTION_MIDDLE_DEVICE4_NB_MAX}
				}},
				choice{ReductionAlgorithm::Block256, 2, {
					condition{0, 1, CUMAT_REDUCTION_MIDDLE_BLOCK256_THRESH},
					condition{-1, 0, -CUMAT_REDUCTION_MIDDLE_BLOCK256_NB_MAX}
				}},
				choice{ReductionAlgorithm::Warp, 2, {
					condition{0, 1, CUMAT_REDUCTION_MIDDLE_WARP_THRESH},
					condition{-1, 0, -CUMAT_REDUCTION_MIDDLE_WARP_NB_MAX}
				}}
			};
			static const ReductionAlgorithm DEFAULT = ReductionAlgorithm::Thread;
			return select(CONDITIONS, DEFAULT, numBatches, batchSize);
		}

		//--- Outer axis (batch reduction) ---
#ifndef CUMAT_REDUCTION_OUTER_DEVICE1_THRESH
#define CUMAT_REDUCTION_OUTER_DEVICE1_THRESH 19.0
#endif
#ifndef CUMAT_REDUCTION_OUTER_DEVICE1_NB_MIN
#define CUMAT_REDUCTION_OUTER_DEVICE1_NB_MIN 2.0
#endif
#ifndef CUMAT_REDUCTION_OUTER_DEVICE4_THRESH
#define CUMAT_REDUCTION_OUTER_DEVICE4_THRESH 184.25
#endif
#ifndef CUMAT_REDUCTION_OUTER_DEVICE4_NB_MIN
#define CUMAT_REDUCTION_OUTER_DEVICE4_NB_MIN 2.0
#endif
#ifndef CUMAT_REDUCTION_OUTER_DEVICE4_NB_MAX
#define CUMAT_REDUCTION_OUTER_DEVICE4_NB_MAX 4.25
#endif
#ifndef CUMAT_REDUCTION_OUTER_DEVICE2_THRESH
#define CUMAT_REDUCTION_OUTER_DEVICE2_THRESH 14.085555
#endif
#ifndef CUMAT_REDUCTION_OUTER_DEVICE2_NB_MIN
#define CUMAT_REDUCTION_OUTER_DEVICE2_NB_MIN 2.0
#endif
#ifndef CUMAT_REDUCTION_OUTER_DEVICE2_NB_MAX
#define CUMAT_REDUCTION_OUTER_DEVICE2_NB_MAX 4.25
#endif
#ifndef CUMAT_REDUCTION_OUTER_SEGMENTED_THRESH
#define CUMAT_REDUCTION_OUTER_SEGMENTED_THRESH 11.5
#endif
#ifndef CUMAT_REDUCTION_OUTER_SEGMENTED_NB_MIN
#define CUMAT_REDUCTION_OUTER_SEGMENTED_NB_MIN 4.0
#endif
#ifndef CUMAT_REDUCTION_OUTER_SEGMENTED_BS_CAP
#define CUMAT_REDUCTION_OUTER_SEGMENTED_BS_CAP 8.5
#endif
#ifndef CUMAT_REDUCTION_OUTER_BLOCK256_THRESH
#define CUMAT_REDUCTION_OUTER_BLOCK256_THRESH 8.0
#endif
#ifndef CUMAT_REDUCTION_OUTER_BLOCK256_NB_MAX
#define CUMAT_REDUCTION_OUTER_BLOCK256_NB_MAX 2.0
#endif
#ifndef CUMAT_REDUCTION_OUTER_WARP_THRESH
#define CUMAT_REDUCTION_OUTER_WARP_THRESH 2.75
#endif
#ifndef CUMAT_REDUCTION_OUTER_WARP_NB_MAX
#define CUMAT_REDUCTION_OUTER_WARP_NB_MAX 11.75
#endif

		static ReductionAlgorithm outer(Index numBatches, Index batchSize)
		{
			static const choice CONDITIONS[] = {
				choice{ReductionAlgorithm::Device1, 2, {
					condition{-1, 0, -CUMAT_REDUCTION_OUTER_DEVICE1_NB_MIN},
					condition{1.875, 1, CUMAT_REDUCTION_OUTER_DEVICE1_THRESH}
				}},
				choice{ReductionAlgorithm::Device4, 3, {
					condition{1, 0, CUMAT_REDUCTION_OUTER_DEVICE4_NB_MIN},
					condition{-1, 0, -CUMAT_REDUCTION_OUTER_DEVICE4_NB_MAX},
					condition{10.0 / 9.0, 1, CUMAT_REDUCTION_OUTER_DEVICE4_THRESH}
				}},
				choice{ReductionAlgorithm::Device2, 3, {
					condition{1, 0, CUMAT_REDUCTION_OUTER_DEVICE2_NB_MIN},
					condition{-1, 0, -CUMAT_REDUCTION_OUTER_DEVICE2_NB_MAX},
					condition{-0.22222, 1, CUMAT_REDUCTION_OUTER_DEVICE2_THRESH}
				}},
				choice{ReductionAlgorithm::Segmented, 3, {
					condition{1, 0, CUMAT_REDUCTION_OUTER_SEGMENTED_NB_MIN},
					condition{0, 1, CUMAT_REDUCTION_OUTER_SEGMENTED_THRESH},
					condition{-1, 0, -CUMAT_REDUCTION_OUTER_SEGMENTED_BS_CAP}
				}},
				choice{ReductionAlgorithm::Block256, 2, {
					condition{0, 1, CUMAT_REDUCTION_OUTER_BLOCK256_THRESH},
					condition{-1, 0, -CUMAT_REDUCTION_OUTER_BLOCK256_NB_MAX}
				}},
				choice{ReductionAlgorithm::Warp, 2, {
					condition{0, 1, CUMAT_REDUCTION_OUTER_WARP_THRESH},
					condition{-1, 0, -CUMAT_REDUCTION_OUTER_WARP_NB_MAX}
				}}
			};
			static const ReductionAlgorithm DEFAULT = ReductionAlgorithm::Thread;
			return select(CONDITIONS, DEFAULT, numBatches, batchSize);
		}
	};
}

CUMAT_NAMESPACE_END

#endif