#include "Utils.h"
#include <cuMat/src/Iterator.h>
using namespace cuMat;
using Index3 = thrust::tuple<Index,Index,Index>;

// Forward declare our test struct for the kernel
struct FromLinearTestCase { Index3 dims; Index3 stride; int total; int idx; char name[32]; };

// Reference: the original (linear / stride) % dims formula
__host__ __device__ Index3 refFromLinear(Index linear, const Index3& dims, const Index3& stride)
{
    return thrust::make_tuple(
        (linear / thrust::get<0>(stride)) % thrust::get<0>(dims),
        (linear / thrust::get<1>(stride)) % thrust::get<1>(dims),
        (linear / thrust::get<2>(stride)) % thrust::get<2>(dims));
}

// The optimized version
__host__ __device__ Index3 optFromLinear(Index linear, const Index3& dims, const Index3& stride)
{
    Index s0 = thrust::get<0>(stride);
    Index s1 = thrust::get<1>(stride);
    Index s2 = thrust::get<2>(stride);
    Index d0 = thrust::get<0>(dims);
    Index d1 = thrust::get<1>(dims);
    Index d2 = thrust::get<2>(dims);
    Index c0, c1, c2;
    if (s0 == s1 || s0 == s2 || s1 == s2)
    {
        c0 = (linear / s0) % d0;
        c1 = (linear / s1) % d1;
        c2 = (linear / s2) % d2;
    }
    else if (s2 >= s0 && s2 >= s1)
    {
        c2 = linear / s2; linear -= c2 * s2;
        if (s0 > s1) {
            c0 = linear / s0; c1 = linear - c0 * s0;
        } else {
            c1 = linear / s1; c0 = linear - c1 * s1;
        }
    } else if (s1 >= s0 && s1 >= s2) {
        c1 = linear / s1; linear -= c1 * s1;
        if (s0 > s2) {
            c0 = linear / s0; c2 = linear - c0 * s0;
        } else {
            c2 = linear / s2; c0 = linear - c2 * s2;
        }
    } else {
        c0 = linear / s0; linear -= c0 * s0;
        if (s1 > s2) {
            c1 = linear / s1; c2 = linear - c1 * s1;
        } else {
            c2 = linear / s2; c1 = linear - c2 * s2;
        }
    }
    return thrust::make_tuple(c0, c1, c2);
}

// GPU kernel: compare StridedMatrixInputIterator::fromLinear vs refFromLinear
template <typename Derived>
__global__ void compareStaticMethodKernel(const FromLinearTestCase* cases, int numCases, int* errors)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCases = 0;
    for (int c = 0; c < numCases; ++c) totalCases += cases[c].total;
    if (tid >= totalCases) return;

    int lin = tid;
    int caseIdx = 0;
    while (caseIdx < numCases && lin >= cases[caseIdx].total) {
        lin -= cases[caseIdx].total;
        ++caseIdx;
    }
    if (caseIdx >= numCases) return;

    auto& cas = cases[caseIdx];
    Index3 ref = refFromLinear(static_cast<Index>(lin), cas.dims, cas.stride);
    Index3 opt = StridedMatrixInputIterator<Derived>::fromLinear(static_cast<Index>(lin), cas.dims, cas.stride);

    bool ok = (thrust::get<0>(ref) == thrust::get<0>(opt)) &&
              (thrust::get<1>(ref) == thrust::get<1>(opt)) &&
              (thrust::get<2>(ref) == thrust::get<2>(opt));
    if (!ok)
        atomicAdd(errors, 1);
}

// GPU kernel: compare standalone optFromLinear vs refFromLinear
__global__ void compareStandaloneKernel(const FromLinearTestCase* cases, int numCases, int* errors)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCases = 0;
    for (int c = 0; c < numCases; ++c) totalCases += cases[c].total;
    if (tid >= totalCases) return;

    int lin = tid;
    int caseIdx = 0;
    while (caseIdx < numCases && lin >= cases[caseIdx].total) {
        lin -= cases[caseIdx].total;
        ++caseIdx;
    }
    if (caseIdx >= numCases) return;

    auto& cas = cases[caseIdx];
    Index3 ref = refFromLinear(static_cast<Index>(lin), cas.dims, cas.stride);
    Index3 opt = optFromLinear(static_cast<Index>(lin), cas.dims, cas.stride);

    bool ok = (thrust::get<0>(ref) == thrust::get<0>(opt)) &&
              (thrust::get<1>(ref) == thrust::get<1>(opt)) &&
              (thrust::get<2>(ref) == thrust::get<2>(opt));
    if (!ok)
        atomicAdd(errors, 1);
}

TEST(FromLinearTest, GPUKernelCompare)
{
    struct Case { Index3 dims; Index3 stride; const char* name; };
    const Case cases[] = {
        {{2,3,1}, {1,2,6},   "Row(2x3x1)"},
        {{2,3,1}, {3,1,6},   "Col(2x3x1)"},
        {{2,3,2}, {1,2,6},   "Row(2x3x2)"},
        {{2,3,2}, {3,1,6},   "Col(2x3x2)"},
        {{2,3,2}, {2,4,1},   "Batch(2x3x2)"},
        {{5,7,3}, {1,5,35},  "Row(5x7x3)"},
        {{5,7,3}, {7,1,35},  "Col(5x7x3)"},
        {{5,7,3}, {3,15,1},  "Batch(5x7x3)"},
        {{1,1,1}, {1,1,1},   "1x1x1"},
        {{4,1,3}, {1,4,4},   "Row(4x1x3)"},
        {{1,5,2}, {5,1,5},   "Col(1x5x2)"},
        {{1,3,1}, {1,1,3},   "OutRow(1x3x1)"},
        {{2,1,1}, {1,1,2},   "OutCol(2x1x1)"},
        {{1,1,3}, {1,1,1},   "OutBatch(1x1x3)"},
        {{3,5,1}, {1,3,15},  "Row(3x5x1)"},
        {{3,5,1}, {5,1,15},  "Col(3x5x1)"},
    };
    int numCases = sizeof(cases) / sizeof(cases[0]);

    int totalElements = 0;
    for (auto& c : cases) {
        totalElements += thrust::get<0>(c.dims) * thrust::get<1>(c.dims) * thrust::get<2>(c.dims);
    }
    std::vector<FromLinearTestCase> hostCases(numCases);
    for (int i = 0; i < numCases; ++i) {
        hostCases[i].dims = cases[i].dims;
        hostCases[i].stride = cases[i].stride;
        hostCases[i].total = static_cast<int>(
            thrust::get<0>(cases[i].dims) * thrust::get<1>(cases[i].dims) * thrust::get<2>(cases[i].dims));
        hostCases[i].idx = i;
        strncpy(hostCases[i].name, cases[i].name, 31);
        hostCases[i].name[31] = 0;
    }

    FromLinearTestCase* d_cases;
    int* d_errors;
    CUMAT_SAFE_CALL(cudaMalloc(&d_cases, numCases * sizeof(FromLinearTestCase)));
    CUMAT_SAFE_CALL(cudaMalloc(&d_errors, sizeof(int)));
    CUMAT_SAFE_CALL(cudaMemcpy(d_cases, hostCases.data(), numCases * sizeof(FromLinearTestCase), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (totalElements + threads - 1) / threads;

    // Test 1: standalone optFromLinear (should pass)
    CUMAT_SAFE_CALL(cudaMemset(d_errors, 0, sizeof(int)));
    compareStandaloneKernel<<<blocks, threads>>>(d_cases, numCases, d_errors);
    CUMAT_SAFE_CALL(cudaDeviceSynchronize());
    int hostErrors = 0;
    CUMAT_SAFE_CALL(cudaMemcpy(&hostErrors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(0, hostErrors) << "Standalone optFromLinear on GPU has mismatches";

    // Test 2: StridedMatrixInputIterator::fromLinear static method
    CUMAT_SAFE_CALL(cudaMemset(d_errors, 0, sizeof(int)));
    compareStaticMethodKernel<MatrixXdR><<<blocks, threads>>>(d_cases, numCases, d_errors);
    CUMAT_SAFE_CALL(cudaDeviceSynchronize());
    hostErrors = 0;
    CUMAT_SAFE_CALL(cudaMemcpy(&hostErrors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(0, hostErrors) << "StridedMatrixInputIterator::fromLinear on GPU has mismatches";

    CUMAT_SAFE_CALL(cudaFree(d_cases));
    CUMAT_SAFE_CALL(cudaFree(d_errors));
}

TEST(FromLinearTest, CompareOptimization)
{
    struct Case { Index3 dims; Index3 stride; const char* name; };
    const Case cases[] = {
        // Reduction iterator strides
        {{2,3,1}, {1,2,6},   "Row(2x3x1)"},
        {{2,3,1}, {3,1,6},   "Col(2x3x1)"},
        {{2,3,2}, {1,2,6},   "Row(2x3x2)"},
        {{2,3,2}, {3,1,6},   "Col(2x3x2)"},
        {{2,3,2}, {2,4,1},   "Batch(2x3x2)"},
        {{5,7,3}, {1,5,35},  "Row(5x7x3)"},
        {{5,7,3}, {7,1,35},  "Col(5x7x3)"},
        {{5,7,3}, {3,15,1},  "Batch(5x7x3)"},
        {{1,1,1}, {1,1,1},   "1x1x1"},
        {{4,1,3}, {1,4,4},   "Row(4x1x3)"},
        {{1,5,2}, {5,1,5},   "Col(1x5x2)"},
        // Output iterator strides
        {{1,3,1}, {1,1,3},   "OutRow(1x3x1)"},
        {{2,1,1}, {1,1,2},   "OutCol(2x1x1)"},
        {{1,1,3}, {1,1,1},   "OutBatch(1x1x3)"},
        // Non-square, single batch
        {{3,5,1}, {1,3,15},  "Row(3x5x1)"},
        {{3,5,1}, {5,1,15},  "Col(3x5x1)"},
    };
    for (auto& c : cases) {
        Index total = thrust::get<0>(c.dims) * thrust::get<1>(c.dims) * thrust::get<2>(c.dims);
        for (Index lin = 0; lin < total; ++lin) {
            Index3 ref = refFromLinear(lin, c.dims, c.stride);
            Index3 opt = optFromLinear(lin, c.dims, c.stride);
            ASSERT_EQ(thrust::get<0>(ref), thrust::get<0>(opt))
                << c.name << " FAIL lin=" << lin << " c0 ref=" << thrust::get<0>(ref) << " opt=" << thrust::get<0>(opt);
            ASSERT_EQ(thrust::get<1>(ref), thrust::get<1>(opt))
                << c.name << " FAIL lin=" << lin << " c1 ref=" << thrust::get<1>(ref) << " opt=" << thrust::get<1>(opt);
            ASSERT_EQ(thrust::get<2>(ref), thrust::get<2>(opt))
                << c.name << " FAIL lin=" << lin << " c2 ref=" << thrust::get<2>(ref) << " opt=" << thrust::get<2>(opt);
        }
    }
}
