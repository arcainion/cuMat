#include "Utils.h"

using namespace cuMat;
using cuMat::CSR;

TEST(CGTest, SimpleSolve)
{
    double aData[1][3][3] = {{
        {5, 2, 1},
        {2, 6, 1},
        {1, 1, 7}
    }};
    double bData[1][3][1] = {{
        {8},
        {9},
        {9}
    }};
    auto A = MatrixXdR::fromArray(aData);
    auto b = MatrixXdR::fromArray(bData);

    ConjugateGradient<decltype(A), DiagonalPreconditioner<decltype(A)>> cg(A);
    cg.setTolerance(1e-8);
    cg.setMaxIterations(100);
    auto x = cg.solve(b).eval();
    auto check = (A * x).eval();
    EXPECT_TRUE(MatrixNear(check, b, 1e-6));
}

TEST(CGTest, IdentityPreconditioner)
{
    double aData[1][3][3] = {{
        {5, 2, 1},
        {2, 6, 1},
        {1, 1, 7}
    }};
    double bData[1][3][1] = {{
        {8},
        {9},
        {9}
    }};
    auto A = MatrixXdR::fromArray(aData);
    auto b = MatrixXdR::fromArray(bData);

    ConjugateGradient<decltype(A), IdentityPreconditioner<decltype(A)>> cg(A);
    cg.setTolerance(1e-8);
    cg.setMaxIterations(200);
    auto x = cg.solve(b).eval();
    auto check = (A * x).eval();
    EXPECT_TRUE(MatrixNear(check, b, 1e-6));
}

TEST(CGTest, SparseCSRSolve)
{
    std::vector<int> rowPtr = {0, 2, 4, 6};
    std::vector<int> colInd = {0, 1, 0, 1, 1, 2};
    std::vector<double> values = {4.0, 1.0, 1.0, 4.0, 1.0, 4.0};
    SparsityPattern<CSR> pattern;
    pattern.rows = 3;
    pattern.cols = 3;
    pattern.nnz = static_cast<int>(colInd.size());
    pattern.JA = Matrix<int, Dynamic, 1, 1, ColumnMajor>(4, 1, 1);
    pattern.IA = Matrix<int, Dynamic, 1, 1, ColumnMajor>(6, 1, 1);
    pattern.JA.copyFromHost(rowPtr.data());
    pattern.IA.copyFromHost(colInd.data());

    SMatrixXd A(pattern);
    A.getData().copyFromHost(values.data());

    double bData[1][3][1] = {{ {6}, {6}, {6} }};
    auto b = MatrixXdR::fromArray(bData);

    ConjugateGradient<SMatrixXd, DiagonalPreconditioner<SMatrixXd>> cg(A);
    cg.setTolerance(1e-8);
    cg.setMaxIterations(100);
    auto x = cg.solve(b).eval();
    auto check = (A * x).eval();
    EXPECT_TRUE(MatrixNear(check, b, 1e-6));
}

TEST(CGTest, ZeroInitialGuess)
{
    double aData[1][2][2] = {{
        {4, 1},
        {1, 3}
    }};
    double bData[1][2][1] = {{
        {5},
        {4}
    }};
    auto A = MatrixXdR::fromArray(aData);
    auto b = MatrixXdR::fromArray(bData);

    ConjugateGradient<decltype(A), DiagonalPreconditioner<decltype(A)>> cg(A);
    cg.setTolerance(1e-10);
    cg.setMaxIterations(50);
    auto x = cg.solve(b).eval();
    auto check = (A * x).eval();
    EXPECT_TRUE(MatrixNear(check, b, 1e-8));
}


