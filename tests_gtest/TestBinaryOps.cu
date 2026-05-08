#include "Utils.h"

using namespace cuMat;

TEST(BinaryOpTest, Add)
{
    float a[1][2][2] = {{{1, 2}, {3, 4}}};
    float b[1][2][2] = {{{5, 6}, {7, 8}}};
    MatrixXfR mA = MatrixXfR::fromArray(a);
    MatrixXfR mB = MatrixXfR::fromArray(b);
    MatrixXfR result = (mA + mB).eval();
    float expected[1][2][2] = {{{6, 8}, {10, 12}}};
    EXPECT_TRUE(MatrixNear(result, MatrixXfR::fromArray(expected), 1e-6));
}

TEST(BinaryOpTest, Subtract)
{
    float a[1][2][2] = {{{5, 6}, {7, 8}}};
    float b[1][2][2] = {{{1, 2}, {3, 4}}};
    MatrixXfR mA = MatrixXfR::fromArray(a);
    MatrixXfR mB = MatrixXfR::fromArray(b);
    MatrixXfR result = (mA - mB).eval();
    float expected[1][2][2] = {{{4, 4}, {4, 4}}};
    EXPECT_TRUE(MatrixNear(result, MatrixXfR::fromArray(expected), 1e-6));
}

TEST(BinaryOpTest, ScalarMultiply)
{
    float a[1][2][2] = {{{1, 2}, {3, 4}}};
    MatrixXfR mA = MatrixXfR::fromArray(a);
    MatrixXfR result = (mA * 3.0f).eval();
    float expected[1][2][2] = {{{3, 6}, {9, 12}}};
    EXPECT_TRUE(MatrixNear(result, MatrixXfR::fromArray(expected), 1e-6));
}

TEST(BinaryOpTest, ScalarDivide)
{
    float a[1][2][2] = {{{2, 4}, {6, 8}}};
    MatrixXfR mA = MatrixXfR::fromArray(a);
    MatrixXfR result = (mA / 2.0f).eval();
    float expected[1][2][2] = {{{1, 2}, {3, 4}}};
    EXPECT_TRUE(MatrixNear(result, MatrixXfR::fromArray(expected), 1e-6));
}

TEST(BinaryOpTest, CwiseMul)
{
    float a[1][2][2] = {{{1, 2}, {3, 4}}};
    float b[1][2][2] = {{{2, 3}, {4, 5}}};
    MatrixXfR mA = MatrixXfR::fromArray(a);
    MatrixXfR mB = MatrixXfR::fromArray(b);
    MatrixXfR result = mA.cwiseMul(mB).eval();
    float expected[1][2][2] = {{{2, 6}, {12, 20}}};
    EXPECT_TRUE(MatrixNear(result, MatrixXfR::fromArray(expected), 1e-6));
}

TEST(BinaryOpTest, CwiseDiv)
{
    float a[1][2][2] = {{{2, 6}, {12, 20}}};
    float b[1][2][2] = {{{2, 3}, {4, 5}}};
    MatrixXfR mA = MatrixXfR::fromArray(a);
    MatrixXfR mB = MatrixXfR::fromArray(b);
    MatrixXfR result = mA.cwiseDiv(mB).eval();
    float expected[1][2][2] = {{{1, 2}, {3, 4}}};
    EXPECT_TRUE(MatrixNear(result, MatrixXfR::fromArray(expected), 1e-6));
}

TEST(BinaryOpTest, CwisePow)
{
    float a[1][1][3] = {{{2, 3, 4}}};
    float b[1][1][3] = {{{2, 2, 2}}};
    MatrixXfR mA = MatrixXfR::fromArray(a);
    MatrixXfR mB = MatrixXfR::fromArray(b);
    auto result = mA.cwisePow(mB).eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(4.0f, host[0]);
    EXPECT_FLOAT_EQ(9.0f, host[1]);
    EXPECT_FLOAT_EQ(16.0f, host[2]);
}

TEST(BinaryOpTest, CwiseEqual)
{
    float a[1][2][2] = {{{1, 2}, {3, 4}}};
    float b[1][2][2] = {{{1, 0}, {3, 5}}};
    MatrixXfR mA = MatrixXfR::fromArray(a);
    MatrixXfR mB = MatrixXfR::fromArray(b);
    auto result = (mA == mB).eval();
    int expected[1][2][2] = {{{1, 0}, {1, 0}}};
    EXPECT_TRUE(MatrixNear(result.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(BinaryOpTest, CwiseLess)
{
    float a[1][2][2] = {{{1, 2}, {3, 4}}};
    float b[1][2][2] = {{{2, 2}, {2, 5}}};
    MatrixXfR mA = MatrixXfR::fromArray(a);
    MatrixXfR mB = MatrixXfR::fromArray(b);
    auto result = (mA < mB).eval();
    int expected[1][2][2] = {{{1, 0}, {0, 1}}};
    EXPECT_TRUE(MatrixNear(result.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(BinaryOpTest, CwiseGreater)
{
    float a[1][2][2] = {{{2, 1}, {3, 4}}};
    float b[1][2][2] = {{{1, 2}, {3, 5}}};
    MatrixXfR mA = MatrixXfR::fromArray(a);
    MatrixXfR mB = MatrixXfR::fromArray(b);
    auto result = (mA > mB).eval();
    int expected[1][2][2] = {{{1, 0}, {0, 0}}};
    EXPECT_TRUE(MatrixNear(result.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(BinaryOpTest, CwiseLessEq)
{
    float a[1][2][2] = {{{1, 2}, {3, 4}}};
    float b[1][2][2] = {{{1, 1}, {3, 5}}};
    MatrixXfR mA = MatrixXfR::fromArray(a);
    MatrixXfR mB = MatrixXfR::fromArray(b);
    auto result = (mA <= mB).eval();
    int expected[1][2][2] = {{{1, 0}, {1, 1}}};
    EXPECT_TRUE(MatrixNear(result.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(BinaryOpTest, CwiseGreaterEq)
{
    float a[1][2][2] = {{{1, 2}, {3, 4}}};
    float b[1][2][2] = {{{1, 3}, {3, 2}}};
    MatrixXfR mA = MatrixXfR::fromArray(a);
    MatrixXfR mB = MatrixXfR::fromArray(b);
    auto result = (mA >= mB).eval();
    int expected[1][2][2] = {{{1, 0}, {1, 1}}};
    EXPECT_TRUE(MatrixNear(result.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(BinaryOpTest, IntegerModulo)
{
    int a[1][2][2] = {{{5, 7}, {9, 11}}};
    int b[1][2][2] = {{{2, 3}, {4, 5}}};
    MatrixXiR mA = MatrixXiR::fromArray(a);
    MatrixXiR mB = MatrixXiR::fromArray(b);
    auto result = (mA % mB).eval();
    int expected[1][2][2] = {{{1, 1}, {1, 1}}};
    EXPECT_TRUE(MatrixNear(result.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(BinaryOpTest, ScalarMultiplyInplace)
{
    MatrixXf m = MatrixXf::Constant(3, 3, 1, 2.0f);
    MatrixXf expected = MatrixXf::Constant(3, 3, 1, 6.0f);
    m *= 3.0f;
    EXPECT_TRUE(MatrixNear(m, expected, 1e-6));
}

TEST(BinaryOpTest, AddInplace)
{
    MatrixXf a = MatrixXf::Constant(2, 2, 1, 3.0f);
    MatrixXf b = MatrixXf::Constant(2, 2, 1, 4.0f);
    a += b.eval();
    MatrixXf expected = MatrixXf::Constant(2, 2, 1, 7.0f);
    EXPECT_TRUE(MatrixNear(a, expected, 1e-6));
}

TEST(BinaryOpTest, SubtractInplace)
{
    MatrixXf a = MatrixXf::Constant(2, 2, 1, 10.0f);
    MatrixXf b = MatrixXf::Constant(2, 2, 1, 3.0f);
    a -= b.eval();
    MatrixXf expected = MatrixXf::Constant(2, 2, 1, 7.0f);
    EXPECT_TRUE(MatrixNear(a, expected, 1e-6));
}

TEST(BinaryOpTest, DivideInplace)
{
    MatrixXf a = MatrixXf::Constant(2, 2, 1, 10.0f);
    a = (a / 2.0f).eval();
    MatrixXf expected = MatrixXf::Constant(2, 2, 1, 5.0f);
    EXPECT_TRUE(MatrixNear(a, expected, 1e-6));
}

TEST(BinaryOpTest, BroadcastAdd)
{
    int mData[1][2][4] = {{
        {1, 2, 6, 9},
        {3, 1, 7, 2}
    }};
    int vData[1][2][1] = {{
        {0},
        {1}
    }};
    auto m = MatrixXiR::fromArray(mData);
    auto v = MatrixXiR::fromArray(vData);
    auto result = (m + v).eval();
    int expected[1][2][4] = {{
        {1, 2, 6, 9},
        {4, 2, 8, 3}
    }};
    EXPECT_TRUE(MatrixNear(result.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(BinaryOpTest, IntegerBitwiseAnd)
{
    int a[1][2][2] = {{{5, 6}, {7, 8}}};
    int b[1][2][2] = {{{3, 3}, {3, 3}}};
    MatrixXiR mA = MatrixXiR::fromArray(a);
    MatrixXiR mB = MatrixXiR::fromArray(b);
    auto result = mA.cwiseBinaryAnd(mB).eval();
    int expected[1][2][2] = {{{1, 2}, {3, 0}}};
    EXPECT_TRUE(MatrixNear(result.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(BinaryOpTest, IntegerBitwiseOr)
{
    int a[1][2][2] = {{{1, 2}, {4, 8}}};
    int b[1][2][2] = {{{2, 4}, {8, 16}}};
    MatrixXiR mA = MatrixXiR::fromArray(a);
    MatrixXiR mB = MatrixXiR::fromArray(b);
    auto result = mA.cwiseBinaryOr(mB).eval();
    int expected[1][2][2] = {{{3, 6}, {12, 24}}};
    EXPECT_TRUE(MatrixNear(result.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(BinaryOpTest, IntegerBitwiseXor)
{
    int a[1][2][2] = {{{1, 2}, {4, 8}}};
    int b[1][2][2] = {{{1, 2}, {4, 8}}};
    MatrixXiR mA = MatrixXiR::fromArray(a);
    MatrixXiR mB = MatrixXiR::fromArray(b);
    auto result = mA.cwiseBinaryXor(mB).eval();
    int expected[1][2][2] = {{{0, 0}, {0, 0}}};
    EXPECT_TRUE(MatrixNear(result.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(BinaryOpTest, DoublePrecisionAdd)
{
    double a[1][2][2] = {{{1.5, 2.5}, {3.5, 4.5}}};
    double b[1][2][2] = {{{0.5, 1.5}, {2.5, 3.5}}};
    MatrixXdR mA = MatrixXdR::fromArray(a);
    MatrixXdR mB = MatrixXdR::fromArray(b);
    MatrixXdR result = (mA + mB).eval();
    double expected[1][2][2] = {{{2.0, 4.0}, {6.0, 8.0}}};
    EXPECT_TRUE(MatrixNear(result, MatrixXdR::fromArray(expected), 1e-12));
}

TEST(BinaryOpTest, CompoundDivide)
{
    MatrixXf a = MatrixXf::Constant(2, 2, 1, 10.0f);
    MatrixXf b = MatrixXf::Constant(2, 2, 1, 2.0f);
    a /= b.eval();
    MatrixXf expected = MatrixXf::Constant(2, 2, 1, 5.0f);
    EXPECT_TRUE(MatrixNear(a, expected, 1e-6));
}

TEST(BinaryOpTest, CompoundModulo)
{
    int aData[1][2][2] = {{{5, 7}, {9, 11}}};
    int bData[1][2][2] = {{{2, 3}, {4, 5}}};
    MatrixXiR a = MatrixXiR::fromArray(aData);
    MatrixXiR b = MatrixXiR::fromArray(bData);
    a %= b.eval();
    int expected[1][2][2] = {{{1, 1}, {1, 1}}};
    EXPECT_TRUE(MatrixNear(a.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(BinaryOpTest, CompoundBitwiseAnd)
{
    int aData[1][2][2] = {{{5, 6}, {7, 8}}};
    int bData[1][2][2] = {{{3, 3}, {3, 3}}};
    MatrixXiR a = MatrixXiR::fromArray(aData);
    MatrixXiR b = MatrixXiR::fromArray(bData);
    a &= b.eval();
    int expected[1][2][2] = {{{1, 2}, {3, 0}}};
    EXPECT_TRUE(MatrixNear(a.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(BinaryOpTest, CompoundBitwiseOr)
{
    int aData[1][2][2] = {{{1, 2}, {4, 8}}};
    int bData[1][2][2] = {{{2, 4}, {8, 16}}};
    MatrixXiR a = MatrixXiR::fromArray(aData);
    MatrixXiR b = MatrixXiR::fromArray(bData);
    a |= b.eval();
    int expected[1][2][2] = {{{3, 6}, {12, 24}}};
    EXPECT_TRUE(MatrixNear(a.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(BinaryOpTest, CompoundMatrixMultiply)
{
    float aData[1][2][2] = {{{1, 2}, {3, 4}}};
    float bData[1][2][2] = {{{2, 0}, {0, 2}}};
    MatrixXfR a = MatrixXfR::fromArray(aData);
    MatrixXfR b = MatrixXfR::fromArray(bData);
    a *= b.eval();
    float expected[1][2][2] = {{{2, 4}, {6, 8}}};
    EXPECT_TRUE(MatrixNear(a, MatrixXfR::fromArray(expected), 1e-6));
}
