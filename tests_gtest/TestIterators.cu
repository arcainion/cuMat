#include "Utils.h"

using namespace cuMat;

template<typename MatrixType>
__global__ void testIteratorKernel(
    MatrixType m,
    float* output, int n)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
    {
        output[i] = m.getRawCoeff(i);
    }
}

TEST(IteratorTest, RowMajorIteration)
{
    float data[1][2][3] = {{
        {1, 2, 3},
        {4, 5, 6}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);

    std::vector<float> expected = {1, 2, 3, 4, 5, 6};
    std::vector<float> result(6);
    float* d_result;
    CUMAT_SAFE_CALL(cudaMalloc(&d_result, 6 * sizeof(float)));
    testIteratorKernel<<<1, 64>>>(m, d_result, 6);
    CUMAT_SAFE_CALL(cudaMemcpy(result.data(), d_result, 6 * sizeof(float), cudaMemcpyDeviceToHost));
    CUMAT_SAFE_CALL(cudaFree(d_result));

    for (int i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(expected[i], result[i]);
}

TEST(IteratorTest, DefaultColumnMajorIteration)
{
    MatrixXf m(2, 3, 1);
    float expectedHost[6] = {1, 4, 2, 5, 3, 6};
    m.copyFromHost(expectedHost);

    std::vector<float> result(6);
    float* d_result;
    CUMAT_SAFE_CALL(cudaMalloc(&d_result, 6 * sizeof(float)));
    testIteratorKernel<<<1, 64>>>(m, d_result, 6);
    CUMAT_SAFE_CALL(cudaMemcpy(result.data(), d_result, 6 * sizeof(float), cudaMemcpyDeviceToHost));
    CUMAT_SAFE_CALL(cudaFree(d_result));

    EXPECT_FLOAT_EQ(1, result[0]);
    EXPECT_FLOAT_EQ(4, result[1]);
    EXPECT_FLOAT_EQ(2, result[2]);
}

TEST(IteratorTest, BatchedIteration)
{
    float data[2][2][2]{{
        {1, 2},
        {3, 4}
    }, {
        {5, 6},
        {7, 8}
    }};
    BMatrixXfR m = BMatrixXfR::fromArray(data);

    std::vector<float> expected = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> result(8);
    float* d_result;
    CUMAT_SAFE_CALL(cudaMalloc(&d_result, 8 * sizeof(float)));
    testIteratorKernel<<<1, 64>>>(m, d_result, 8);
    CUMAT_SAFE_CALL(cudaMemcpy(result.data(), d_result, 8 * sizeof(float), cudaMemcpyDeviceToHost));
    CUMAT_SAFE_CALL(cudaFree(d_result));

    for (int i = 0; i < 8; ++i)
        EXPECT_FLOAT_EQ(expected[i], result[i]);
}
