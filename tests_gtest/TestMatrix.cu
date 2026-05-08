#include "Utils.h"

using namespace cuMat;

TEST(MatrixTest, DefaultConstructor)
{
    MatrixXf m;
    EXPECT_EQ(0, m.rows());
    EXPECT_EQ(0, m.cols());
    EXPECT_EQ(1, m.batches());
    EXPECT_EQ(0, m.size());
}

TEST(MatrixTest, SizeConstructor)
{
    MatrixXf m(3, 4, 1);
    EXPECT_EQ(3, m.rows());
    EXPECT_EQ(4, m.cols());
    EXPECT_EQ(1, m.batches());
    EXPECT_EQ(12, m.size());
}

TEST(MatrixTest, FixedSizeConstructor)
{
    Matrix3f m;
    EXPECT_EQ(3, m.rows());
    EXPECT_EQ(3, m.cols());
    EXPECT_EQ(1, m.batches());
    EXPECT_EQ(9, m.size());
}

TEST(MatrixTest, BatchedConstructor)
{
    BMatrixXf m(2, 3, 5);
    EXPECT_EQ(2, m.rows());
    EXPECT_EQ(3, m.cols());
    EXPECT_EQ(5, m.batches());
    EXPECT_EQ(30, m.size());
}

TEST(MatrixTest, FromArray)
{
    float data[1][2][3] = {{
        {1, 2, 3},
        {4, 5, 6}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    EXPECT_EQ(2, m.rows());
    EXPECT_EQ(3, m.cols());
    EXPECT_EQ(1, m.batches());
}

TEST(MatrixTest, ConstantFill)
{
    MatrixXf m = MatrixXf::Constant(2, 2, 1, 3.14f);
    EXPECT_EQ(2, m.rows());
    EXPECT_EQ(2, m.cols());
    MatrixXf expected = MatrixXf::Constant(2, 2, 1, 3.14f);
    EXPECT_TRUE(MatrixNear(m, expected, 1e-6));
}

TEST(MatrixTest, ZeroFill)
{
    MatrixXf m = MatrixXf::Zero(4, 3, 1);
    MatrixXf expected = MatrixXf::Constant(4, 3, 1, 0.0f);
    EXPECT_TRUE(MatrixNear(m, expected, 1e-6));
}

TEST(MatrixTest, IdentityFill)
{
    MatrixXf m = MatrixXf::Identity(3, 3, 1);
    auto eval = m.eval();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            float expectedVal = (i == j) ? 1.0f : 0.0f;
            std::vector<float> host(1);
            eval.block(i, j, 0, 1, 1, 1).eval().copyToHost(host.data());
            EXPECT_FLOAT_EQ(expectedVal, host[0]);
        }
    }
}

TEST(MatrixTest, CopyConstructor)
{
    MatrixXf a = MatrixXf::Constant(3, 3, 1, 5.0f);
    MatrixXf b(a);
    EXPECT_TRUE(MatrixNear(a, b, 1e-6));
}

TEST(MatrixTest, AssignmentOperator)
{
    MatrixXf a = MatrixXf::Constant(3, 3, 1, 7.0f);
    MatrixXf b;
    b = a;
    EXPECT_TRUE(MatrixNear(a, b, 1e-6));
}

TEST(MatrixTest, DeepClone)
{
    MatrixXf a = MatrixXf::Constant(3, 3, 1, 9.0f);
    MatrixXf b = a.deepClone();
    EXPECT_TRUE(MatrixNear(a, b, 1e-6));

    a.setZero();
    MatrixXf expected = MatrixXf::Constant(3, 3, 1, 9.0f);
    EXPECT_TRUE(MatrixNear(b, expected, 1e-6));
}

TEST(MatrixTest, SetZero)
{
    MatrixXf a = MatrixXf::Constant(4, 5, 1, 3.0f);
    a.setZero();
    MatrixXf expected = MatrixXf::Zero(4, 5, 1);
    EXPECT_TRUE(MatrixNear(a, expected, 1e-6));
}

TEST(MatrixTest, BlockOperation)
{
    MatrixXf a = MatrixXf::Constant(5, 5, 1, 1.0f);
    auto block = a.block(1, 1, 0, 2, 2, 1).eval();

    MatrixXf expected = MatrixXf::Constant(2, 2, 1, 1.0f);
    EXPECT_TRUE(MatrixNear(block, expected, 1e-6));
}

TEST(MatrixTest, BlockLvalueAssignment)
{
    MatrixXf a = MatrixXf::Zero(5, 5, 1);
    MatrixXf blockVal = MatrixXf::Constant(2, 2, 1, 3.0f);
    a.block(0, 0, 0, 2, 2, 1) = blockVal.eval();

    std::vector<float> host(1);
    a.block(0, 0, 0, 1, 1, 1).eval().copyToHost(host.data());
    EXPECT_FLOAT_EQ(3.0f, host[0]);
    a.block(3, 3, 0, 1, 1, 1).eval().copyToHost(host.data());
    EXPECT_FLOAT_EQ(0.0f, host[0]);
}

TEST(MatrixTest, RowAccess)
{
    MatrixXf a(3, 3, 1);
    SimpleRandom rnd(123);
    rnd.fillUniform(a, 0.0f, 1.0f);

    auto row1 = a.row(1);
    EXPECT_EQ(1, row1.rows());
    EXPECT_EQ(3, row1.cols());
}

TEST(MatrixTest, ColAccess)
{
    MatrixXf a(3, 3, 1);
    SimpleRandom rnd(456);
    rnd.fillUniform(a, 0.0f, 1.0f);

    auto col1 = a.col(1);
    EXPECT_EQ(3, col1.rows());
    EXPECT_EQ(1, col1.cols());
}

TEST(MatrixTest, CopyFromHost)
{
    std::vector<float> host = {1.0f, 2.0f, 3.0f, 4.0f};
    MatrixXf a(2, 2, 1);
    a.copyFromHost(host.data());

    std::vector<float> out(4);
    a.copyToHost(out.data());
    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(host[i], out[i]);
}

TEST(MatrixTest, CopyToHost)
{
    MatrixXf a = MatrixXf::Constant(2, 3, 1, 2.5f);
    std::vector<float> host(6);
    a.copyToHost(host.data());
    for (float v : host)
        EXPECT_FLOAT_EQ(2.5f, v);
}

TEST(MatrixTest, RandomFill)
{
    MatrixXf a(10, 10, 1);
    SimpleRandom rnd(42);
    rnd.fillUniform(a, 0.0f, 1.0f);

    std::vector<float> host(100);
    a.copyToHost(host.data());
    for (float v : host)
    {
        EXPECT_GE(v, 0.0f);
        EXPECT_LT(v, 1.0f);
    }
}
