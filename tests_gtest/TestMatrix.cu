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

TEST(MatrixTest, EmptyMatrix)
{
    MatrixXf m;
    EXPECT_EQ(0, m.rows());
    EXPECT_EQ(0, m.cols());
    EXPECT_EQ(1, m.batches());
    EXPECT_EQ(0, m.size());
    MatrixXf zero = MatrixXf::Zero(0, 0, 1);
    EXPECT_EQ(0, zero.size());
    auto sum = zero.sum();
    EXPECT_FLOAT_EQ(0.0f, static_cast<float>(sum));
}

TEST(MatrixTest, SingleElementMatrix)
{
    MatrixXf m = MatrixXf::Constant(1, 1, 1, 42.0f);
    EXPECT_EQ(1, m.rows());
    EXPECT_EQ(1, m.cols());
    std::vector<float> host(1);
    m.copyToHost(host.data());
    EXPECT_FLOAT_EQ(42.0f, host[0]);
    auto sum = m.sum();
    EXPECT_FLOAT_EQ(42.0f, static_cast<float>(sum));
    auto prod = m.prod();
    EXPECT_FLOAT_EQ(42.0f, static_cast<float>(prod));
}

TEST(MatrixTest, MatrixOfZeros)
{
    MatrixXf m = MatrixXf::Zero(4, 5, 1);
    std::vector<float> host(20);
    m.copyToHost(host.data());
    for (float v : host)
        EXPECT_FLOAT_EQ(0.0f, v);
}

TEST(MatrixTest, ScalarMultiplication)
{
    MatrixXf a = MatrixXf::Constant(2, 2, 1, 2.0f);
    MatrixXf result = (a * 3.0f).eval();
    std::vector<float> host(4);
    result.copyToHost(host.data());
    for (float v : host)
        EXPECT_FLOAT_EQ(6.0f, v);
}

TEST(MatrixTest, ZeroSizedBatch)
{
    BMatrixXf m(2, 3, 0);
    EXPECT_EQ(0, m.batches());
    EXPECT_EQ(0, m.size());
}

TEST(MatrixTest, Blas1Axpy)
{
    MatrixXf a = MatrixXf::Constant(2, 2, 1, 1.0f);
    MatrixXf b = MatrixXf::Constant(2, 2, 1, 2.0f);
    a.axpy(3.0f, b);
    std::vector<float> host(4);
    a.copyToHost(host.data());
    for (float v : host)
        EXPECT_FLOAT_EQ(7.0f, v);
}

TEST(MatrixTest, Blas1Copy)
{
    MatrixXf a = MatrixXf::Constant(2, 2, 1, 1.0f);
    MatrixXf b = MatrixXf::Constant(2, 2, 1, 2.0f);
    a.copy(b);
    std::vector<float> host(4);
    a.copyToHost(host.data());
    for (float v : host)
        EXPECT_FLOAT_EQ(2.0f, v);
}

TEST(MatrixTest, Blas1Scal)
{
    MatrixXf a = MatrixXf::Constant(2, 2, 1, 3.0f);
    a.scal(2.0f);
    std::vector<float> host(4);
    a.copyToHost(host.data());
    for (float v : host)
        EXPECT_FLOAT_EQ(6.0f, v);
}

TEST(MatrixTest, Slice)
{
    BMatrixXfR m(2, 3, 2);
    float hostData[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    m.copyFromHost(hostData);

    auto s0 = m.slice(0).eval();
    EXPECT_EQ(2, s0.rows());
    EXPECT_EQ(3, s0.cols());
    EXPECT_EQ(1, s0.batches());
    std::vector<float> host(6);
    s0.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1, host[0]);
    EXPECT_FLOAT_EQ(2, host[1]);
    EXPECT_FLOAT_EQ(3, host[2]);
    EXPECT_FLOAT_EQ(4, host[3]);
    EXPECT_FLOAT_EQ(5, host[4]);
    EXPECT_FLOAT_EQ(6, host[5]);

    auto s1 = m.slice(1).eval();
    s1.copyToHost(host.data());
    EXPECT_FLOAT_EQ(7, host[0]);
    EXPECT_FLOAT_EQ(8, host[1]);
    EXPECT_FLOAT_EQ(9, host[2]);
    EXPECT_FLOAT_EQ(10, host[3]);
    EXPECT_FLOAT_EQ(11, host[4]);
    EXPECT_FLOAT_EQ(12, host[5]);
}

TEST(MatrixTest, Segment)
{
    float data[1][5][1] = {{{1}, {2}, {3}, {4}, {5}}};
    auto v = VectorXfR::fromArray(data);

    auto seg = v.segment<3>(1).eval();
    EXPECT_EQ(3, seg.rows());
    std::vector<float> host(3);
    seg.copyToHost(host.data());
    EXPECT_FLOAT_EQ(2, host[0]);
    EXPECT_FLOAT_EQ(3, host[1]);
    EXPECT_FLOAT_EQ(4, host[2]);

    auto segDyn = v.segment(2, 2).eval();
    EXPECT_EQ(2, segDyn.rows());
    segDyn.copyToHost(host.data());
    EXPECT_FLOAT_EQ(3, host[0]);
    EXPECT_FLOAT_EQ(4, host[1]);
}

TEST(MatrixTest, Head)
{
    float data[1][5][1] = {{{1}, {2}, {3}, {4}, {5}}};
    auto v = VectorXfR::fromArray(data);

    auto h = v.head<3>().eval();
    EXPECT_EQ(3, h.rows());
    std::vector<float> host(3);
    h.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1, host[0]);
    EXPECT_FLOAT_EQ(2, host[1]);
    EXPECT_FLOAT_EQ(3, host[2]);
}

TEST(MatrixTest, Tail)
{
    float data[1][5][1] = {{{1}, {2}, {3}, {4}, {5}}};
    auto v = VectorXfR::fromArray(data);

    auto t = v.tail<3>().eval();
    EXPECT_EQ(3, t.rows());
    std::vector<float> host(3);
    t.copyToHost(host.data());
    EXPECT_FLOAT_EQ(3, host[0]);
    EXPECT_FLOAT_EQ(4, host[1]);
    EXPECT_FLOAT_EQ(5, host[2]);
}

struct SquareFunctor
{
    typedef float ReturnType;
    __device__ CUMAT_STRONG_INLINE float operator()(const float& v, Index row, Index col, Index batch) const
    {
        return v * v;
    }
};

TEST(MatrixTest, UnaryExpr)
{
    MatrixXfR m = MatrixXfR::Constant(2, 2, 1, 3.0f);
    auto result = m.unaryExpr(SquareFunctor()).eval();
    std::vector<float> host(4);
    result.copyToHost(host.data());
    for (float v : host)
        EXPECT_FLOAT_EQ(9.0f, v);
}

struct AddDoubleFunctor
{
    typedef float ReturnType;
    __device__ CUMAT_STRONG_INLINE float operator()(const float& a, const float& b, Index row, Index col, Index batch) const
    {
        return a + 2.0f * b;
    }
};

TEST(MatrixTest, BinaryExpr)
{
    MatrixXfR a = MatrixXfR::Constant(2, 2, 1, 1.0f);
    MatrixXfR b = MatrixXfR::Constant(2, 2, 1, 2.0f);
    auto result = a.binaryExpr(b, AddDoubleFunctor()).eval();
    std::vector<float> host(4);
    result.copyToHost(host.data());
    for (float v : host)
        EXPECT_FLOAT_EQ(5.0f, v);
}

struct RowColSumFunctor
{
    typedef float ReturnType;
    __device__ CUMAT_STRONG_INLINE float operator()(Index row, Index col, Index batch) const
    {
        return static_cast<float>(row + col);
    }
};

TEST(MatrixTest, NullaryExpr)
{
    auto m = MatrixXfR::NullaryExpr(2, 3, 1, RowColSumFunctor()).eval();
    EXPECT_EQ(2, m.rows());
    EXPECT_EQ(3, m.cols());
    std::vector<float> host(6);
    m.copyToHost(host.data());
    EXPECT_FLOAT_EQ(0, host[0]);
    EXPECT_FLOAT_EQ(1, host[1]);
    EXPECT_FLOAT_EQ(2, host[2]);
    EXPECT_FLOAT_EQ(1, host[3]);
    EXPECT_FLOAT_EQ(2, host[4]);
    EXPECT_FLOAT_EQ(3, host[5]);
}
