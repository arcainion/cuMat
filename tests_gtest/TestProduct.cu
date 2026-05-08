#include "Utils.h"

using namespace cuMat;

TEST(ProductTest, MatrixVectorProduct)
{
    float aData[1][2][3] = {{
        {1, 2, 3},
        {4, 5, 6}
    }};
    float vData[1][3][1] = {{
        {7},
        {8},
        {9}
    }};
    MatrixXfR A = MatrixXfR::fromArray(aData);
    MatrixXfR v = MatrixXfR::fromArray(vData);
    auto result = (A * v).eval();
    EXPECT_EQ(2, result.rows());
    EXPECT_EQ(1, result.cols());

    std::vector<float> host(2);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(50.0f, host[0]);
    EXPECT_FLOAT_EQ(122.0f, host[1]);
}

TEST(ProductTest, MatrixMatrixProduct)
{
    float aData[1][2][3] = {{
        {1, 2, 3},
        {4, 5, 6}
    }};
    float bData[1][3][2] = {{
        {7, 8},
        {9, 10},
        {11, 12}
    }};
    MatrixXfR A = MatrixXfR::fromArray(aData);
    MatrixXfR B = MatrixXfR::fromArray(bData);
    auto result = (A * B).eval();
    EXPECT_EQ(2, result.rows());
    EXPECT_EQ(2, result.cols());

    std::vector<float> host(4);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(58.0f, host[0]);
    EXPECT_FLOAT_EQ(139.0f, host[1]);
    EXPECT_FLOAT_EQ(64.0f, host[2]);
    EXPECT_FLOAT_EQ(154.0f, host[3]);
}

TEST(ProductTest, SquareMatrixProduct)
{
    float aData[1][3][3] = {{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    }};
    float bData[1][3][3] = {{
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    }};
    MatrixXfR A = MatrixXfR::fromArray(aData);
    MatrixXfR B = MatrixXfR::fromArray(bData);
    auto result = (A * B).eval();
    EXPECT_EQ(3, result.rows());
    EXPECT_EQ(3, result.cols());

    std::vector<float> host(9);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(30.0f, host[0]);
    EXPECT_FLOAT_EQ(24.0f, host[3]);
    EXPECT_FLOAT_EQ(18.0f, host[6]);
}

TEST(ProductTest, TransposeProduct)
{
    float aData[1][2][3] = {{
        {1, 2, 3},
        {4, 5, 6}
    }};
    MatrixXfR A = MatrixXfR::fromArray(aData);
    auto result = (A.transpose() * A).eval();
    EXPECT_EQ(3, result.rows());
    EXPECT_EQ(3, result.cols());

    std::vector<float> host(9);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(17.0f, host[0]);
    EXPECT_FLOAT_EQ(22.0f, host[1]);
    EXPECT_FLOAT_EQ(27.0f, host[2]);
}

TEST(ProductTest, IdentityProduct)
{
    MatrixXf A = MatrixXf::Constant(3, 3, 1, 5.0f);
    MatrixXf I = MatrixXf::Identity(3, 3, 1);
    auto result = (A * I).eval();
    EXPECT_TRUE(MatrixNear(result, A, 1e-6));
}

TEST(ProductTest, BatchedProduct)
{
    float aData[2][2][3] = {{
        {1, 2, 3},
        {4, 5, 6}
    }, {
        {7, 8, 9},
        {10, 11, 12}
    }};
    float bData[2][3][2] = {{
        {1, 0},
        {0, 1},
        {0, 0}
    }, {
        {0, 0},
        {1, 0},
        {0, 1}
    }};
    BMatrixXfR A = BMatrixXfR::fromArray(aData);
    BMatrixXfR B = BMatrixXfR::fromArray(bData);
    auto result = (A * B).eval();
    EXPECT_EQ(2, result.rows());
    EXPECT_EQ(2, result.cols());
    EXPECT_EQ(2, result.batches());

    std::vector<float> host(8);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1.0f, host[0]);
    EXPECT_FLOAT_EQ(4.0f, host[1]);
    EXPECT_FLOAT_EQ(2.0f, host[2]);
    EXPECT_FLOAT_EQ(5.0f, host[3]);
    EXPECT_FLOAT_EQ(8.0f, host[4]);
    EXPECT_FLOAT_EQ(11.0f, host[5]);
    EXPECT_FLOAT_EQ(9.0f, host[6]);
    EXPECT_FLOAT_EQ(12.0f, host[7]);
}

TEST(ProductTest, DoublePrecisionProduct)
{
    double aData[1][2][2] = {{
        {1.5, 2.5},
        {3.5, 4.5}
    }};
    double bData[1][2][2] = {{
        {0.5, 1.5},
        {2.5, 3.5}
    }};
    MatrixXdR A = MatrixXdR::fromArray(aData);
    MatrixXdR B = MatrixXdR::fromArray(bData);
    auto result = (A * B).eval();

    std::vector<double> host(4);
    result.copyToHost(host.data());
    EXPECT_DOUBLE_EQ(7.0, host[0]);
    EXPECT_DOUBLE_EQ(13.0, host[1]);
    EXPECT_DOUBLE_EQ(11.0, host[2]);
    EXPECT_DOUBLE_EQ(21.0, host[3]);
}

TEST(ProductTest, VectorDotViaProduct)
{
    float aData[1][3][1] = {{{1}, {2}, {3}}};
    float bData[1][1][3] = {{{4, 5, 6}}};
    auto a = MatrixXfR::fromArray(aData);
    auto b = MatrixXfR::fromArray(bData);
    auto result = (a.transpose() * b.transpose()).eval();
    EXPECT_EQ(1, result.rows());
    EXPECT_EQ(1, result.cols());
    std::vector<float> host(1);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(32.0f, host[0]);
}
