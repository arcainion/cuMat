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

TEST(SparseTest, CSCMatrixVectorProduct)
{
    // Matrix A (3x3, CSC):
    // [1 2 0]
    // [0 3 0]
    // [4 0 5]
    //
    // Column 0: rows 0,2 -> values 1,4
    // Column 1: row 1     -> value  3
    // Column 2: rows 0,2 -> values 2,5
    //
    // JA (col ptrs): [0, 2, 3, 5]
    // IA (row idxs): [0, 2, 1, 0, 2]
    // Values:        [1, 4, 3, 2, 5]

    std::vector<int> colPtr = {0, 2, 3, 5};
    std::vector<int> rowInd = {0, 2, 1, 0, 2};
    std::vector<float> values = {1.0f, 4.0f, 3.0f, 2.0f, 5.0f};

    SparsityPattern<CSC> pattern;
    pattern.rows = 3;
    pattern.cols = 3;
    pattern.nnz = 5;
    pattern.JA = Matrix<int, Dynamic, 1, 1, ColumnMajor>(4, 1, 1);
    pattern.IA = Matrix<int, Dynamic, 1, 1, ColumnMajor>(5, 1, 1);
    pattern.JA.copyFromHost(colPtr.data());
    pattern.IA.copyFromHost(rowInd.data());

    SMatrixXf_CSC A(pattern);
    A.getData().copyFromHost(values.data());

    std::vector<float> vecData = {1.0f, 2.0f, 3.0f};
    VectorXf v(3, 1, 1);
    v.copyFromHost(vecData.data());

    auto result = (A * v).eval();
    EXPECT_EQ(3, result.rows());

    std::vector<float> host(3);
    result.copyToHost(host.data());
    // y[0] = 1*1 + 2*3 = 7
    // y[1] = 3*2 = 6
    // y[2] = 4*1 + 5*3 = 19
    EXPECT_FLOAT_EQ(7.0f, host[0]);
    EXPECT_FLOAT_EQ(6.0f, host[1]);
    EXPECT_FLOAT_EQ(19.0f, host[2]);
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

TEST(SparseTest, CSRMatrixMatrixProduct)
{
    // CSR matrix A (3×3):
    // [1 2 0]
    // [0 3 0]
    // [4 0 5]
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

    // Dense matrix B (3×2):
    // [1 4]
    // [2 5]
    // [3 6]
    float bData[6] = {1, 2, 3, 4, 5, 6};
    MatrixXfR B(3, 2, 1);
    B.copyFromHost(bData);

    // C = A * B (3×2)
    auto C = (A * B).eval();
    EXPECT_EQ(3, C.rows());
    EXPECT_EQ(2, C.cols());

    // B is row-major 3×2: [1 2; 3 4; 5 6]
    // C = A * B, output is column-major:
    // C[0][0] = 1*1 + 2*3 = 7
    // C[1][0] = 3*3 = 9
    // C[2][0] = 4*1 + 5*5 = 29
    // C[0][1] = 1*2 + 2*4 = 10
    // C[1][1] = 3*4 = 12
    // C[2][1] = 4*2 + 5*6 = 38
    std::vector<float> host(6);
    C.copyToHost(host.data());
    EXPECT_FLOAT_EQ(7.0f, host[0]);
    EXPECT_FLOAT_EQ(9.0f, host[1]);
    EXPECT_FLOAT_EQ(29.0f, host[2]);
    EXPECT_FLOAT_EQ(10.0f, host[3]);
    EXPECT_FLOAT_EQ(12.0f, host[4]);
    EXPECT_FLOAT_EQ(38.0f, host[5]);
}

TEST(SparseTest, ELLPACKMatrixVectorProduct)
{
    // Matrix A (3x3, ELLPACK, nnzPerRow=2):
    // [1 0 2]
    // [0 3 0]
    // [4 0 5]

    SparsityPattern<ELLPACK> pattern;
    pattern.rows = 3;
    pattern.cols = 3;
    pattern.nnzPerRow = 2;

    // indices matrix (3x2, column-major)
    std::vector<int> indicesData = {
        0, 1, 0,
        2, -1, 2
    };
    pattern.indices = Matrix<int, Dynamic, Dynamic, 1, ColumnMajor>(3, 2, 1);
    pattern.indices.copyFromHost(indicesData.data());

    SMatrixXf_ELLPACK A(pattern);
    std::vector<float> valuesData = {
        1.0f, 3.0f, 4.0f,
        2.0f, 0.0f, 5.0f
    };
    A.getData().copyFromHost(valuesData.data());

    std::vector<float> vecData = {1.0f, 1.0f, 1.0f};
    VectorXf v(3, 1, 1);
    v.copyFromHost(vecData.data());

    auto result = (A * v).eval();
    EXPECT_EQ(3, result.rows());

    std::vector<float> host(3);
    result.copyToHost(host.data());
    EXPECT_FLOAT_EQ(3.0f, host[0]);
    EXPECT_FLOAT_EQ(3.0f, host[1]);
    EXPECT_FLOAT_EQ(9.0f, host[2]);
}
