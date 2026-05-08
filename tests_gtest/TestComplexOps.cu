#include "Utils.h"

using namespace cuMat;

TEST(ComplexTest, Construction)
{
    cfloat data[1][2][2] = {{
        {cfloat(1, 2), cfloat(3, 4)},
        {cfloat(5, 6), cfloat(7, 8)}
    }};
    MatrixXcfR m = MatrixXcfR::fromArray(data);
    EXPECT_EQ(2, m.rows());
    EXPECT_EQ(2, m.cols());
}

TEST(ComplexTest, RealImagExtraction)
{
    cfloat data[1][2][2] = {{
        {cfloat(1, 2), cfloat(3, 4)},
        {cfloat(5, 6), cfloat(7, 8)}
    }};
    MatrixXcfR m = MatrixXcfR::fromArray(data);

    auto realPart = m.real().eval();
    auto imagPart = m.imag().eval();

    std::vector<float> realHost(4);
    realPart.copyToHost(realHost.data());
    EXPECT_FLOAT_EQ(1, realHost[0]);
    EXPECT_FLOAT_EQ(3, realHost[1]);
    EXPECT_FLOAT_EQ(5, realHost[2]);
    EXPECT_FLOAT_EQ(7, realHost[3]);

    std::vector<float> imagHost(4);
    imagPart.copyToHost(imagHost.data());
    EXPECT_FLOAT_EQ(2, imagHost[0]);
    EXPECT_FLOAT_EQ(4, imagHost[1]);
    EXPECT_FLOAT_EQ(6, imagHost[2]);
    EXPECT_FLOAT_EQ(8, imagHost[3]);
}

TEST(ComplexTest, ComplexAddition)
{
    cfloat data1[1][2][2] = {{
        {cfloat(1, 2), cfloat(3, 4)},
        {cfloat(5, 6), cfloat(7, 8)}
    }};
    cfloat data2[1][2][2] = {{
        {cfloat(8, 7), cfloat(6, 5)},
        {cfloat(4, 3), cfloat(2, 1)}
    }};
    MatrixXcfR a = MatrixXcfR::fromArray(data1);
    MatrixXcfR b = MatrixXcfR::fromArray(data2);
    auto result = (a + b).eval();

    cfloat expected[1][2][2] = {{
        {cfloat(9, 9), cfloat(9, 9)},
        {cfloat(9, 9), cfloat(9, 9)}
    }};
    EXPECT_TRUE(MatrixNear(result, MatrixXcfR::fromArray(expected), 1e-6));
}

TEST(ComplexTest, Conjugate)
{
    cfloat data[1][1][3] = {{
        {cfloat(1, 1), cfloat(2, -2), cfloat(3, 0)}
    }};
    MatrixXcfR m = MatrixXcfR::fromArray(data);
    auto result = m.conjugate().eval();

    std::vector<cfloat> host(3);
    result.copyToHost(host.data());
    EXPECT_EQ(cfloat(1, -1), host[0]);
    EXPECT_EQ(cfloat(2, 2), host[1]);
    EXPECT_EQ(cfloat(3, 0), host[2]);
}

TEST(ComplexTest, Adjoint)
{
    cfloat data[1][2][3] = {{
        {cfloat(1, 1), cfloat(2, 2), cfloat(3, 3)},
        {cfloat(4, 4), cfloat(5, 5), cfloat(6, 6)}
    }};
    MatrixXcfR m = MatrixXcfR::fromArray(data);
    auto adj = m.adjoint().eval();
    EXPECT_EQ(3, adj.rows());
    EXPECT_EQ(2, adj.cols());

    std::vector<cfloat> host(6);
    adj.copyToHost(host.data());
    EXPECT_EQ(cfloat(1, -1), host[0]);
    EXPECT_EQ(cfloat(2, -2), host[1]);
    EXPECT_EQ(cfloat(3, -3), host[2]);
}

TEST(ComplexTest, ComplexNorm)
{
    cfloat data[1][1][2] = {{
        {cfloat(3, 4), cfloat(0, 5)}
    }};
    MatrixXcfR m = MatrixXcfR::fromArray(data);
    float norm = static_cast<float>(m.squaredNorm());
    EXPECT_FLOAT_EQ(50.0f, norm);
}

TEST(ComplexTest, ComplexDotProduct)
{
    cfloat data1[1][1][2] = {{{cfloat(1, 2), cfloat(3, 4)}}};
    cfloat data2[1][1][2] = {{{cfloat(5, 6), cfloat(7, 8)}}};
    auto a = MatrixXcfR::fromArray(data1);
    auto b = MatrixXcfR::fromArray(data2);
    cfloat result = static_cast<cfloat>(a.dot(b));
    EXPECT_EQ(cfloat(-18, 68), result);
}

TEST(ComplexTest, ComplexCwiseAbs)
{
    cfloat data[1][1][3] = {{
        {cfloat(3, 4), cfloat(0, 5), cfloat(1, 0)}
    }};
    MatrixXcfR m = MatrixXcfR::fromArray(data);
    auto result = m.cwiseAbs().eval();

    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(5.0f, host[0]);
    EXPECT_FLOAT_EQ(5.0f, host[1]);
    EXPECT_FLOAT_EQ(1.0f, host[2]);
}

TEST(ComplexTest, ComplexMatrixProduct)
{
    cfloat aData[1][2][2] = {{
        {cfloat(1, 0), cfloat(2, 0)},
        {cfloat(3, 0), cfloat(4, 0)}
    }};
    cfloat bData[1][2][2] = {{
        {cfloat(1, 0), cfloat(0, 0)},
        {cfloat(0, 0), cfloat(1, 0)}
    }};
    MatrixXcfR A = MatrixXcfR::fromArray(aData);
    MatrixXcfR B = MatrixXcfR::fromArray(bData);
    auto result = (A * B).eval();
    EXPECT_TRUE(MatrixNear(result, A, 1e-6));
}

TEST(ComplexTest, DoubleComplex)
{
    cdouble data[1][2][2] = {{
        {cdouble(1.5, 2.5), cdouble(3.5, 4.5)},
        {cdouble(5.5, 6.5), cdouble(7.5, 8.5)}
    }};
    MatrixXcdR m = MatrixXcdR::fromArray(data);
    auto result = (m + m).eval();
    cdouble expected[1][2][2] = {{
        {cdouble(3, 5), cdouble(7, 9)},
        {cdouble(11, 13), cdouble(15, 17)}
    }};
    EXPECT_TRUE(MatrixNear(result, MatrixXcdR::fromArray(expected), 1e-12));
}
