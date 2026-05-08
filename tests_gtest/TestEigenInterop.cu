#include "Utils.h"

#include <Eigen/Dense>

using namespace cuMat;

TEST(EigenInteropTest, ToEigenColumnMajor)
{
    // Column-major: data[i + row*cols] = element at (row, col)
    // For 2x3 matrix: [1 3 5; 2 4 6]
    float data[6] = {1, 2, 3, 4, 5, 6};
    MatrixXf m(2, 3, 1);
    m.copyFromHost(data);
    Eigen::MatrixXf em = m.toEigen();
    EXPECT_EQ(2, em.rows());
    EXPECT_EQ(3, em.cols());
    EXPECT_FLOAT_EQ(1, em(0, 0));
    EXPECT_FLOAT_EQ(3, em(0, 1));
    EXPECT_FLOAT_EQ(5, em(0, 2));
    EXPECT_FLOAT_EQ(2, em(1, 0));
    EXPECT_FLOAT_EQ(4, em(1, 1));
    EXPECT_FLOAT_EQ(6, em(1, 2));
}

TEST(EigenInteropTest, ToEigenRowMajor)
{
    // Row-major: data[i + row*cols] = element at (row, col)
    // For 2x3 matrix: [1 2 3; 4 5 6]
    float data[6] = {1, 2, 3, 4, 5, 6};
    MatrixXfR m(2, 3, 1);
    m.copyFromHost(data);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> em = m.toEigen();
    EXPECT_EQ(2, em.rows());
    EXPECT_EQ(3, em.cols());
    EXPECT_FLOAT_EQ(1, em(0, 0));
    EXPECT_FLOAT_EQ(2, em(0, 1));
    EXPECT_FLOAT_EQ(3, em(0, 2));
    EXPECT_FLOAT_EQ(4, em(1, 0));
    EXPECT_FLOAT_EQ(5, em(1, 1));
    EXPECT_FLOAT_EQ(6, em(1, 2));
}

TEST(EigenInteropTest, ToEigenComplex)
{
    // Column-major complex float, 1x3 vector
    std::complex<float> data[3] = {
        std::complex<float>(1, 2),
        std::complex<float>(3, 4),
        std::complex<float>(5, 6)
    };
    MatrixXcf m(1, 3, 1);
    m.copyFromHost(reinterpret_cast<const cfloat*>(data));
    Eigen::MatrixXcf em = m.toEigen();
    EXPECT_EQ(1, em.rows());
    EXPECT_EQ(3, em.cols());
    EXPECT_EQ(std::complex<float>(1, 2), em(0, 0));
    EXPECT_EQ(std::complex<float>(3, 4), em(0, 1));
    EXPECT_EQ(std::complex<float>(5, 6), em(0, 2));
}

TEST(EigenInteropTest, FromEigenColumnMajor)
{
    Eigen::MatrixXf em(2, 3);
    em << 1, 4, 7, 2, 5, 8;
    // Eigen column-major layout: [1 4 7; 2 5 8]
    // Stored as {1, 2, 4, 5, 7, 8}
    MatrixXf m = MatrixXf::fromEigen(em);
    std::vector<float> host(6);
    m.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1, host[0]);
    EXPECT_FLOAT_EQ(2, host[1]);
    EXPECT_FLOAT_EQ(4, host[2]);
    EXPECT_FLOAT_EQ(5, host[3]);
    EXPECT_FLOAT_EQ(7, host[4]);
    EXPECT_FLOAT_EQ(8, host[5]);
}

TEST(EigenInteropTest, FromEigenRowMajor)
{
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> em(2, 3);
    em << 1, 2, 3, 4, 5, 6;
    // Eigen row-major layout: [1 2 3; 4 5 6]
    // Stored as {1, 2, 3, 4, 5, 6}
    MatrixXfR m = MatrixXfR::fromEigen(em);
    std::vector<float> host(6);
    m.copyToHost(host.data());
    EXPECT_FLOAT_EQ(1, host[0]);
    EXPECT_FLOAT_EQ(2, host[1]);
    EXPECT_FLOAT_EQ(3, host[2]);
    EXPECT_FLOAT_EQ(4, host[3]);
    EXPECT_FLOAT_EQ(5, host[4]);
    EXPECT_FLOAT_EQ(6, host[5]);
}
