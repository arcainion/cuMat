#include "Utils.h"

using namespace cuMat;

TEST(UnaryOpTest, Negate)
{
    float data[1][2][2] = {{{1, -2}, {3, -4}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    MatrixXfR result = (-m).eval();
    MatrixXfR expected = MatrixXfR::Constant(2, 2, 1, 0) - m;
    EXPECT_TRUE(MatrixNear(result, expected.eval(), 1e-6));
}

TEST(UnaryOpTest, Abs)
{
    float data[1][2][2] = {{{-1, 2}, {-3, 4}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseAbs().eval();
    float ref[1][2][2] = {{{1, 2}, {3, 4}}};
    EXPECT_TRUE(MatrixNear(result, MatrixXfR::fromArray(ref), 1e-6));
}

TEST(UnaryOpTest, Abs2)
{
    float data[1][2][2] = {{{-2, 3}, {-4, 5}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseAbs2().eval();
    std::vector<float> host(1);
    result.block(0, 0, 0, 1, 1, 1).eval().copyToHost(host.data());
    EXPECT_FLOAT_EQ(4.0f, host[0]);
}

TEST(UnaryOpTest, SquareRoot)
{
    float data[1][2][2] = {{{4, 9}, {16, 25}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseSqrt().eval();
    float expected[1][2][2] = {{{2, 3}, {4, 5}}};
    EXPECT_TRUE(MatrixNear(result, MatrixXfR::fromArray(expected), 1e-5));
}

TEST(UnaryOpTest, Exponential)
{
    float data[1][1][3] = {{{0.0f, 1.0f, 2.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseExp().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1.0f, host[0]);
    EXPECT_FLOAT_EQ(std::exp(1.0f), host[1]);
    EXPECT_FLOAT_EQ(std::exp(2.0f), host[2]);
}

TEST(UnaryOpTest, NaturalLog)
{
    float data[1][1][3] = {{{1.0f, std::exp(1.0f), std::exp(2.0f)}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseLog().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(0.0f, host[0]);
    EXPECT_FLOAT_EQ(1.0f, host[1]);
    EXPECT_FLOAT_EQ(2.0f, host[2]);
}

TEST(UnaryOpTest, Log1p)
{
    float data[1][1][3] = {{{0.0f, 1.0f, 2.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseLog1p().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::log1p(0.0f), host[0]);
    EXPECT_FLOAT_EQ(std::log1p(1.0f), host[1]);
    EXPECT_FLOAT_EQ(std::log1p(2.0f), host[2]);
}

TEST(UnaryOpTest, Log10)
{
    float data[1][1][3] = {{{1.0f, 10.0f, 100.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseLog10().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(0.0f, host[0]);
    EXPECT_FLOAT_EQ(1.0f, host[1]);
    EXPECT_FLOAT_EQ(2.0f, host[2]);
}

TEST(UnaryOpTest, Sine)
{
    float data[1][1][3] = {{{0.0f, 1.0f, 2.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseSin().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::sin(0.0f), host[0]);
    EXPECT_FLOAT_EQ(std::sin(1.0f), host[1]);
    EXPECT_FLOAT_EQ(std::sin(2.0f), host[2]);
}

TEST(UnaryOpTest, Cosine)
{
    float data[1][1][3] = {{{0.0f, 1.0f, 2.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseCos().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::cos(0.0f), host[0]);
    EXPECT_FLOAT_EQ(std::cos(1.0f), host[1]);
    EXPECT_FLOAT_EQ(std::cos(2.0f), host[2]);
}

TEST(UnaryOpTest, Tangent)
{
    float data[1][1][3] = {{{0.0f, 0.5f, 1.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseTan().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::tan(0.0f), host[0]);
    EXPECT_FLOAT_EQ(std::tan(0.5f), host[1]);
    EXPECT_FLOAT_EQ(std::tan(1.0f), host[2]);
}

TEST(UnaryOpTest, Floor)
{
    float data[1][1][4] = {{{1.1f, 1.9f, -1.1f, -1.9f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseFloor().eval();
    std::vector<float> host(4);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1.0f, host[0]);
    EXPECT_FLOAT_EQ(1.0f, host[1]);
    EXPECT_FLOAT_EQ(-2.0f, host[2]);
    EXPECT_FLOAT_EQ(-2.0f, host[3]);
}

TEST(UnaryOpTest, Ceil)
{
    float data[1][1][4] = {{{1.1f, 1.9f, -1.1f, -1.9f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseCeil().eval();
    std::vector<float> host(4);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(2.0f, host[0]);
    EXPECT_FLOAT_EQ(2.0f, host[1]);
    EXPECT_FLOAT_EQ(-1.0f, host[2]);
    EXPECT_FLOAT_EQ(-1.0f, host[3]);
}

TEST(UnaryOpTest, Round)
{
    float data[1][1][4] = {{{1.1f, 1.9f, -1.1f, -1.9f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseRound().eval();
    std::vector<float> host(4);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1.0f, host[0]);
    EXPECT_FLOAT_EQ(2.0f, host[1]);
    EXPECT_FLOAT_EQ(-1.0f, host[2]);
    EXPECT_FLOAT_EQ(-2.0f, host[3]);
}

TEST(UnaryOpTest, Transpose)
{
    float data[1][2][3] = {{{1, 2, 3}, {4, 5, 6}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.transpose().eval();
    EXPECT_EQ(3, result.rows());
    EXPECT_EQ(2, result.cols());

    std::vector<float> host(6);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1, host[0]);
    EXPECT_FLOAT_EQ(2, host[1]);
    EXPECT_FLOAT_EQ(3, host[2]);
}

TEST(UnaryOpTest, CastToDouble)
{
    float data[1][2][2] = {{{1, 2}, {3, 4}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.template cast<double>().eval();
    std::vector<double> host(4);
    result.copyToHost(host.data());
    EXPECT_DOUBLE_EQ(1.0, host[0]);
    EXPECT_DOUBLE_EQ(2.0, host[1]);
    EXPECT_DOUBLE_EQ(3.0, host[2]);
    EXPECT_DOUBLE_EQ(4.0, host[3]);
}

TEST(UnaryOpTest, Inverse)
{
    float data[1][2][2] = {{{1, 2}, {3, 4}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseInverse().eval();
    std::vector<float> host(4);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1.0f/1.0f, host[0]);
    EXPECT_FLOAT_EQ(1.0f/2.0f, host[1]);
    EXPECT_FLOAT_EQ(1.0f/3.0f, host[2]);
    EXPECT_FLOAT_EQ(1.0f/4.0f, host[3]);
}

TEST(UnaryOpTest, Sinh)
{
    float data[1][1][3] = {{{0.0f, 0.5f, 1.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseSinh().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::sinh(0.0f), host[0]);
    EXPECT_FLOAT_EQ(std::sinh(0.5f), host[1]);
    EXPECT_FLOAT_EQ(std::sinh(1.0f), host[2]);
}

TEST(UnaryOpTest, Cosh)
{
    float data[1][1][3] = {{{0.0f, 0.5f, 1.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseCosh().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::cosh(0.0f), host[0]);
    EXPECT_FLOAT_EQ(std::cosh(0.5f), host[1]);
    EXPECT_FLOAT_EQ(std::cosh(1.0f), host[2]);
}

TEST(UnaryOpTest, Tanh)
{
    float data[1][1][3] = {{{0.0f, 0.5f, 1.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseTanh().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::tanh(0.0f), host[0]);
    EXPECT_FLOAT_EQ(std::tanh(0.5f), host[1]);
    EXPECT_FLOAT_EQ(std::tanh(1.0f), host[2]);
}

TEST(UnaryOpTest, Erf)
{
    float data[1][1][3] = {{{0.0f, 0.5f, 1.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseErf().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::erf(0.0f), host[0]);
    EXPECT_FLOAT_EQ(std::erf(0.5f), host[1]);
    EXPECT_FLOAT_EQ(std::erf(1.0f), host[2]);
}

TEST(UnaryOpTest, Erfc)
{
    float data[1][1][3] = {{{0.0f, 0.5f, 1.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseErfc().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::erfc(0.0f), host[0]);
    EXPECT_FLOAT_EQ(std::erfc(0.5f), host[1]);
    EXPECT_FLOAT_EQ(std::erfc(1.0f), host[2]);
}

TEST(UnaryOpTest, Lgamma)
{
    float data[1][1][3] = {{{0.5f, 1.0f, 2.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseLgamma().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::lgamma(0.5f), host[0]);
    EXPECT_FLOAT_EQ(std::lgamma(1.0f), host[1]);
    EXPECT_FLOAT_EQ(std::lgamma(2.0f), host[2]);
}

TEST(UnaryOpTest, LogicalNot)
{
    float data[1][2][2] = {{{0, 1}, {2, 0}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseLogicalNot().eval();
    int expectedData[1][2][2] = {{{1, 0}, {0, 1}}};
    auto expected = MatrixXiR::fromArray(expectedData);
    auto resultInt = result.template cast<int>().eval();
    EXPECT_TRUE(MatrixNear(resultInt.template cast<float>(), expected.template cast<float>(), 1e-6));
}

TEST(UnaryOpTest, ArcSin)
{
    float data[1][1][3] = {{{0.0f, 0.5f, 1.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseAsin().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::asin(0.0f), host[0]);
    EXPECT_FLOAT_EQ(std::asin(0.5f), host[1]);
    EXPECT_FLOAT_EQ(std::asin(1.0f), host[2]);
}

TEST(UnaryOpTest, ArcCos)
{
    float data[1][1][3] = {{{1.0f, 0.5f, 0.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseAcos().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::acos(1.0f), host[0]);
    EXPECT_FLOAT_EQ(std::acos(0.5f), host[1]);
    EXPECT_FLOAT_EQ(std::acos(0.0f), host[2]);
}

TEST(UnaryOpTest, ArcTan)
{
    float data[1][1][3] = {{{0.0f, 1.0f, 2.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseAtan().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::atan(0.0f), host[0]);
    EXPECT_FLOAT_EQ(std::atan(1.0f), host[1]);
    EXPECT_FLOAT_EQ(std::atan(2.0f), host[2]);
}

TEST(UnaryOpTest, ArcSinh)
{
    float data[1][1][3] = {{{0.0f, 0.5f, 1.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseAsinh().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::asinh(0.0f), host[0]);
    EXPECT_FLOAT_EQ(std::asinh(0.5f), host[1]);
    EXPECT_FLOAT_EQ(std::asinh(1.0f), host[2]);
}

TEST(UnaryOpTest, ArcCosh)
{
    float data[1][1][3] = {{{1.0f, 2.0f, 3.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseAcosh().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::acosh(1.0f), host[0]);
    EXPECT_FLOAT_EQ(std::acosh(2.0f), host[1]);
    EXPECT_FLOAT_EQ(std::acosh(3.0f), host[2]);
}

TEST(UnaryOpTest, ArcTanh)
{
    float data[1][1][3] = {{{0.0f, 0.5f, -0.5f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseAtanh().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(std::atanh(0.0f), host[0]);
    EXPECT_FLOAT_EQ(std::atanh(0.5f), host[1]);
    EXPECT_FLOAT_EQ(std::atanh(-0.5f), host[2]);
}

TEST(UnaryOpTest, Rsqrt)
{
    float data[1][1][3] = {{{1.0f, 4.0f, 9.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseRsqrt().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1.0f / std::sqrt(1.0f), host[0]);
    EXPECT_FLOAT_EQ(1.0f / std::sqrt(4.0f), host[1]);
    EXPECT_FLOAT_EQ(1.0f / std::sqrt(9.0f), host[2]);
}

TEST(UnaryOpTest, Cbrt)
{
    float data[1][1][3] = {{{1.0f, 8.0f, 27.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseCbrt().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1.0f, host[0]);
    EXPECT_FLOAT_EQ(2.0f, host[1]);
    EXPECT_FLOAT_EQ(3.0f, host[2]);
}

TEST(UnaryOpTest, BinaryNot)
{
    int data[1][2][2] = {{{0, 1}, {2, 3}}};
    MatrixXiR m = MatrixXiR::fromArray(data);
    auto result = (~m).eval();
    int expected[1][2][2] = {{{-1, -2}, {-3, -4}}};
    EXPECT_TRUE(MatrixNear(result.template cast<float>(), MatrixXiR::fromArray(expected).template cast<float>(), 1e-6));
}

TEST(UnaryOpTest, DiagonalView)
{
    float data[1][3][3] = {{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    }};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto diag = m.diagonal().eval();
    EXPECT_EQ(3, diag.size());
    std::vector<float> host(3);
    diag.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1, host[0]);
    EXPECT_FLOAT_EQ(5, host[1]);
    EXPECT_FLOAT_EQ(9, host[2]);
}

TEST(UnaryOpTest, AsDiagonal)
{
    float data[1][3][1] = {{{1}, {2}, {3}}};
    auto v = MatrixXfR::fromArray(data);
    auto d = v.asDiagonal().eval();
    EXPECT_EQ(3, d.rows());
    EXPECT_EQ(3, d.cols());
    std::vector<float> host(9);
    d.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1, host[0]);
    EXPECT_FLOAT_EQ(2, host[4]);
    EXPECT_FLOAT_EQ(3, host[8]);
}

TEST(UnaryOpTest, InverseCheck)
{
    float data[1][1][4] = {{{0.0f, 1.0f, 2.0f, 4.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseInverseCheck().eval();
    std::vector<float> host(4);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1.0f, host[0]);
    EXPECT_FLOAT_EQ(1.0f, host[1]);
    EXPECT_FLOAT_EQ(0.5f, host[2]);
    EXPECT_FLOAT_EQ(0.25f, host[3]);
}

TEST(UnaryOpTest, Rcbrt)
{
    float data[1][1][3] = {{{1.0f, 8.0f, 27.0f}}};
    MatrixXfR m = MatrixXfR::fromArray(data);
    auto result = m.cwiseRcbrt().eval();
    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1.0f, host[0]);
    EXPECT_FLOAT_EQ(0.5f, host[1]);
    EXPECT_FLOAT_EQ(1.0f / 3.0f, host[2]);
}
