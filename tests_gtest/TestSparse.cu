#include "Utils.h"

using namespace cuMat;
using cuMat::CSR;
using cuMat::CSC;
using cuMat::ELLPACK;

TEST(SparseTest, CSRConstruction)
{
    std::vector<int> rowPtr = {0, 2, 3, 5};
    std::vector<int> colInd = {0, 1, 1, 0, 2};
    SparsityPattern<CSR> pattern;
    pattern.rows = 3;
    pattern.cols = 3;
    pattern.nnz = 5;
    pattern.JA = Matrix<int, Dynamic, 1, 1, ColumnMajor>(4, 1, 1);
    pattern.IA = Matrix<int, Dynamic, 1, 1, ColumnMajor>(5, 1, 1);
    pattern.JA.copyFromHost(rowPtr.data());
    pattern.IA.copyFromHost(colInd.data());
    EXPECT_EQ(3, pattern.rows);
    EXPECT_EQ(3, pattern.cols);
    EXPECT_EQ(5, pattern.nnz);
}

TEST(SparseTest, CSRMatrixVectorProduct)
{
    std::vector<int> rowPtr = {0, 2, 3, 5};
    std::vector<int> colInd = {0, 1, 1, 0, 2};
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    SparsityPattern<CSR> pattern;
    pattern.rows = 3;
    pattern.cols = 3;
    pattern.nnz = 5;
    pattern.JA = Matrix<int, Dynamic, 1, 1, ColumnMajor>(4, 1, 1);
    pattern.IA = Matrix<int, Dynamic, 1, 1, ColumnMajor>(5, 1, 1);
    pattern.JA.copyFromHost(rowPtr.data());
    pattern.IA.copyFromHost(colInd.data());

    SMatrixXf A(pattern);
    A.getData().copyFromHost(values.data());

    std::vector<float> vecData = {1.0f, 2.0f, 3.0f};
    VectorXf v(3, 1, 1);
    v.copyFromHost(vecData.data());

    auto result = (A * v).eval();
    EXPECT_EQ(3, result.rows());

    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(5.0f, host[0]);
    EXPECT_FLOAT_EQ(6.0f, host[1]);
    EXPECT_FLOAT_EQ(19.0f, host[2]);
}

TEST(SparseTest, CSCConstruction)
{
    SparsityPattern<CSC> pattern;
    pattern.rows = 3;
    pattern.cols = 3;
    pattern.nnz = 5;
    EXPECT_EQ(3, pattern.rows);
    EXPECT_EQ(3, pattern.cols);
}

TEST(SparseTest, ELLPACKConstruction)
{
    SparsityPattern<ELLPACK> pattern;
    pattern.rows = 3;
    pattern.cols = 3;
    pattern.nnzPerRow = 2;
    EXPECT_EQ(3, pattern.rows);
    EXPECT_EQ(3, pattern.cols);
}

TEST(SparseTest, BatchedSparseMatrix)
{
    std::vector<int> rowPtr = {0, 1, 3, 4};
    std::vector<int> colInd = {0, 0, 1, 2};
    SparsityPattern<CSR> pattern;
    pattern.rows = 3;
    pattern.cols = 3;
    pattern.nnz = 4;
    pattern.JA = Matrix<int, Dynamic, 1, 1, ColumnMajor>(4, 1, 1);
    pattern.IA = Matrix<int, Dynamic, 1, 1, ColumnMajor>(4, 1, 1);
    pattern.JA.copyFromHost(rowPtr.data());
    pattern.IA.copyFromHost(colInd.data());

    BSMatrixXf A(pattern, 3);
    EXPECT_EQ(3, A.batches());
    EXPECT_EQ(3, A.rows());
    EXPECT_EQ(3, A.cols());
}
