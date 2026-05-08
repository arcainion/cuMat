#include "Utils.h"

using namespace cuMat;

TEST(ReductionTest, Sum)
{
    float data[1][2][3] = {{
        {1, 2, 3},
        {4, 5, 6}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    float result = static_cast<float>(m.sum());
    EXPECT_FLOAT_EQ(21.0f, result);
}

TEST(ReductionTest, Prod)
{
    float data[1][2][2] = {{
        {2, 3},
        {4, 5}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    float result = static_cast<float>(m.prod());
    EXPECT_FLOAT_EQ(120.0f, result);
}

TEST(ReductionTest, MinCoeff)
{
    float data[1][2][3] = {{
        {5, 3, 8},
        {1, 9, 2}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    float result = static_cast<float>(m.minCoeff());
    EXPECT_FLOAT_EQ(1.0f, result);
}

TEST(ReductionTest, MaxCoeff)
{
    float data[1][2][3] = {{
        {5, 3, 8},
        {1, 9, 2}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    float result = static_cast<float>(m.maxCoeff());
    EXPECT_FLOAT_EQ(9.0f, result);
}

TEST(ReductionTest, Trace)
{
    float data[1][3][3] = {{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    float result = static_cast<float>(m.trace());
    EXPECT_FLOAT_EQ(15.0f, result);
}

TEST(ReductionTest, SquaredNorm)
{
    float data[1][1][3] = {{
        {3, 4, 0}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    float result = static_cast<float>(m.squaredNorm());
    EXPECT_FLOAT_EQ(25.0f, result);
}

TEST(ReductionTest, Norm)
{
    float data[1][1][3] = {{
        {3, 4, 0}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    float result = static_cast<float>(m.norm());
    EXPECT_FLOAT_EQ(5.0f, result);
}

TEST(ReductionTest, DotProduct)
{
    float data1[1][3][1] = {{{1}, {2}, {3}}};
    float data2[1][3][1] = {{{4}, {5}, {6}}};
    VectorXf a = VectorXf::fromArray(data1);
    VectorXf b = VectorXf::fromArray(data2);
    float result = static_cast<float>(a.dot(b));
    EXPECT_FLOAT_EQ(32.0f, result);
}

TEST(ReductionTest, SumAlongRows)
{
    double data[1][2][3] = {{
        {1, 2, 3},
        {4, 5, 6}
    }};
    MatrixXdR m = MatrixXdR::fromArray(data);
    auto result = m.sum<Axis::Row>().eval();
    EXPECT_EQ(1, result.rows());
    EXPECT_EQ(3, result.cols());

    std::vector<double> host(3);
    result.copyToHost(host.data());
    EXPECT_DOUBLE_EQ(5.0, host[0]);
    EXPECT_DOUBLE_EQ(7.0, host[1]);
    EXPECT_DOUBLE_EQ(9.0, host[2]);
}

TEST(ReductionTest, SumAlongColumns)
{
    double data[1][2][3] = {{
        {1, 2, 3},
        {4, 5, 6}
    }};
    MatrixXdR m = MatrixXdR::fromArray(data);
    auto result = m.sum<Axis::Column>().eval();
    EXPECT_EQ(2, result.rows());
    EXPECT_EQ(1, result.cols());

    std::vector<double> host(2);
    result.copyToHost(host.data());
    EXPECT_DOUBLE_EQ(6.0, host[0]);
    EXPECT_DOUBLE_EQ(15.0, host[1]);
}

TEST(ReductionTest, SumAllAxes)
{
    double data[2][2][2] {{
        {1, 2},
        {3, 4}
    }, {
        {5, 6},
        {7, 8}
    }};
    BMatrixXdR m = BMatrixXdR::fromArray(data);
    auto result = m.sum<Axis::Row | Axis::Column | Axis::Batch>().eval();
    EXPECT_EQ(1, result.rows());
    EXPECT_EQ(1, result.cols());

    std::vector<double> host(1);
    result.copyToHost(host.data());
    EXPECT_DOUBLE_EQ(36.0, host[0]);
}

TEST(ReductionTest, SumDynamicAxis)
{
    double data[1][2][3] = {{
        {1, 2, 3},
        {4, 5, 6}
    }};
    MatrixXdR m = MatrixXdR::fromArray(data);
    auto result = m.sum(Axis::Row | Axis::Column).eval();
    EXPECT_EQ(1, result.rows());
    EXPECT_EQ(1, result.cols());
    std::vector<double> host(1);
    result.copyToHost(host.data());
    EXPECT_DOUBLE_EQ(21.0, host[0]);
}

TEST(ReductionTest, BooleanReductions)
{
    float data[1][2][3] = {{
        {0, 1, 2},
        {3, 4, 5}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    EXPECT_TRUE(static_cast<bool>((m > 0).any()));
    EXPECT_FALSE(static_cast<bool>((m > 10).any()));
    EXPECT_FALSE(static_cast<bool>((m > 0).all()));
}

TEST(ReductionTest, BatchedReduction)
{
    double data[2][2][2] {{
        {1, 2},
        {3, 4}
    }, {
        {5, 6},
        {7, 8}
    }};
    BMatrixXdR m = BMatrixXdR::fromArray(data);
    auto result = m.sum<Axis::Row | Axis::Column>().eval();
    EXPECT_EQ(1, result.rows());
    EXPECT_EQ(1, result.cols());

    std::vector<double> host(2);
    result.copyToHost(host.data());
    EXPECT_DOUBLE_EQ(10.0, host[0]);
    EXPECT_DOUBLE_EQ(26.0, host[1]);
}

TEST(ReductionTest, ReductionAlgorithmSelection)
{
    float data[1][2][2] = {{
        {1, 2},
        {3, 4}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    float sumWarp = static_cast<float>(m.sum<Axis::Row | Axis::Column | Axis::Batch, ReductionAlg::Warp>());
    EXPECT_FLOAT_EQ(10.0f, sumWarp);
}

TEST(ReductionTest, IntegerReduction)
{
    int data[1][2][3] = {{
        {1, 2, 3},
        {4, 5, 6}
    }};
    MatrixXiR m = MatrixXiR::fromArray(data);
    int sum = static_cast<int>(m.sum());
    EXPECT_EQ(21, sum);
    int prod = static_cast<int>(m.prod());
    EXPECT_EQ(720, prod);
}

TEST(ReductionTest, ReductionAlgorithmSegmented)
{
    float data[1][2][2] = {{
        {1, 2},
        {3, 4}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    float sum = static_cast<float>(m.sum<Axis::Row | Axis::Column | Axis::Batch, ReductionAlg::Segmented>());
    EXPECT_FLOAT_EQ(10.0f, sum);
    float maxc = static_cast<float>(m.maxCoeff<Axis::Row | Axis::Column | Axis::Batch, ReductionAlg::Segmented>());
    EXPECT_FLOAT_EQ(4.0f, maxc);
}

TEST(ReductionTest, ReductionAlgorithmThread)
{
    float data[1][2][2] = {{
        {1, 2},
        {3, 4}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    float sum = static_cast<float>(m.sum<Axis::Row | Axis::Column | Axis::Batch, ReductionAlg::Thread>());
    EXPECT_FLOAT_EQ(10.0f, sum);
    float maxc = static_cast<float>(m.maxCoeff<Axis::Row | Axis::Column | Axis::Batch, ReductionAlg::Thread>());
    EXPECT_FLOAT_EQ(4.0f, maxc);
}

TEST(ReductionTest, ReductionAlgorithmBlock)
{
    float data[1][2][2] = {{
        {1, 2},
        {3, 4}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    float sum = static_cast<float>(m.sum<Axis::Row | Axis::Column | Axis::Batch, ReductionAlg::Block<256>>());
    EXPECT_FLOAT_EQ(10.0f, sum);
    float maxc = static_cast<float>(m.maxCoeff<Axis::Row | Axis::Column | Axis::Batch, ReductionAlg::Block<256>>());
    EXPECT_FLOAT_EQ(4.0f, maxc);
}

TEST(ReductionTest, ReductionAlgorithmDevice)
{
    float data[1][2][2] = {{
        {1, 2},
        {3, 4}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    float sum = static_cast<float>(m.sum<Axis::Row | Axis::Column | Axis::Batch, ReductionAlg::Device<1>>());
    EXPECT_FLOAT_EQ(10.0f, sum);
    float maxc = static_cast<float>(m.maxCoeff<Axis::Row | Axis::Column | Axis::Batch, ReductionAlg::Device<1>>());
    EXPECT_FLOAT_EQ(4.0f, maxc);
}
