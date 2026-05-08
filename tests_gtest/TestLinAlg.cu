#include "Utils.h"

using namespace cuMat;

TEST(LinAlgTest, Determinant2x2)
{
    float data[1][2][2] = {{
        {1, 2},
        {3, 4}
    }};
    MatrixXfR A = MatrixXfR::fromArray(data);
    float det = static_cast<float>(A.determinant().eval());
    EXPECT_FLOAT_EQ(-2.0f, det);
}

TEST(LinAlgTest, Determinant3x3)
{
    float data[1][3][3] = {{
        {6, 4, 1},
        {4, 5, 1},
        {1, 1, 6}
    }};
    MatrixXfR A = MatrixXfR::fromArray(data);
    float det = static_cast<float>(A.determinant().eval());
    EXPECT_FLOAT_EQ(81.0f, det);
}

TEST(LinAlgTest, Determinant4x4)
{
    float data[1][4][4] = {{
        {1, 0, 0, 0},
        {0, 2, 0, 0},
        {0, 0, 3, 0},
        {0, 0, 0, 4}
    }};
    MatrixXfR A = MatrixXfR::fromArray(data);
    float det = static_cast<float>(A.determinant().eval());
    EXPECT_FLOAT_EQ(24.0f, det);
}

TEST(LinAlgTest, LogDeterminant)
{
    float data[1][3][3] = {{
        {6, 4, 1},
        {4, 5, 1},
        {1, 1, 6}
    }};
    MatrixXfR A = MatrixXfR::fromArray(data);
    float logDet = static_cast<float>(A.logDeterminant().eval());
    EXPECT_FLOAT_EQ(std::log(81.0f), logDet);
}

TEST(LinAlgTest, Inverse2x2)
{
    float data[1][2][2] = {{
        {1, 2},
        {3, 4}
    }};
    MatrixXfR A = MatrixXfR::fromArray(data);
    auto inv = A.inverse().eval();
    auto product = (A * inv).eval();
    EXPECT_TRUE(MatrixNear(product, MatrixXf::Identity(2, 2, 1), 1e-5));
}

TEST(LinAlgTest, Inverse3x3)
{
    float data[1][3][3] = {{
        {6, 4, 1},
        {4, 5, 1},
        {1, 1, 6}
    }};
    MatrixXfR A = MatrixXfR::fromArray(data);
    auto inv = A.inverse().eval();
    auto product = (A * inv).eval();
    EXPECT_TRUE(MatrixNear(product, MatrixXf::Identity(3, 3, 1), 1e-5));
}

TEST(LinAlgTest, Inverse4x4)
{
    float data[1][4][4] = {{
        {2, 0, 0, 0},
        {0, 3, 0, 0},
        {0, 0, 4, 0},
        {0, 0, 0, 5}
    }};
    MatrixXfR A = MatrixXfR::fromArray(data);
    auto inv = A.inverse().eval();
    auto product = (A * inv).eval();
    EXPECT_TRUE(MatrixNear(product, MatrixXf::Identity(4, 4, 1), 1e-5));
}

TEST(LinAlgTest, LUSolve2x2)
{
    float aData[1][2][2] = {{
        {1, 2},
        {3, 4}
    }};
    float bData[1][2][1] = {{
        {5},
        {11}
    }};
    MatrixXfR A = MatrixXfR::fromArray(aData);
    MatrixXfR b = MatrixXfR::fromArray(bData);
    auto LU = A.decompositionLU();
    auto x = LU.solve(b).eval();
    std::vector<float> host(2);
    x.copyToHost(host.data());
    EXPECT_NEAR(1.0f, host[0], 1e-4);
    EXPECT_NEAR(2.0f, host[1], 1e-4);

    auto check = (A * x).eval();
    EXPECT_TRUE(MatrixNear(check, b, 1e-4));
}

TEST(LinAlgTest, LUSolve3x3)
{
    float aData[1][3][3] = {{
        {6, 4, 1},
        {4, 5, 1},
        {1, 1, 6}
    }};
    float bData[1][3][1] = {{
        {11},
        {10},
        {8}
    }};
    MatrixXfR A = MatrixXfR::fromArray(aData);
    MatrixXfR b = MatrixXfR::fromArray(bData);
    auto LU = A.decompositionLU();
    auto x = LU.solve(b).eval();
    auto check = (A * x).eval();
    EXPECT_TRUE(MatrixNear(check, b, 1e-4));
}

TEST(LinAlgTest, CholeskySolve2x2)
{
    double aData[1][2][2] = {{
        {4, 1},
        {1, 3}
    }};
    double bData[1][2][1] = {{
        {5},
        {4}
    }};
    MatrixXdR A = MatrixXdR::fromArray(aData);
    MatrixXdR b = MatrixXdR::fromArray(bData);
    auto chol = A.decompositionCholesky();
    auto x = chol.solve(b).eval();
    auto check = (A * x).eval();
    EXPECT_TRUE(MatrixNear(check, b, 1e-10));
}

TEST(LinAlgTest, CholeskySolve3x3)
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
    MatrixXdR A = MatrixXdR::fromArray(aData);
    MatrixXdR b = MatrixXdR::fromArray(bData);
    auto chol = A.decompositionCholesky();
    auto x = chol.solve(b).eval();
    auto check = (A * x).eval();
    EXPECT_TRUE(MatrixNear(check, b, 1e-10));
}

TEST(LinAlgTest, LUDynamicSize)
{
    int n = 5;
    MatrixXf A = MatrixXf::Zero(n, n, 1);
    for (int i = 0; i < n; ++i)
    {
        A.block(i, i, 0, 1, 1, 1) = MatrixXf::Constant(1, 1, 1, static_cast<float>(i + 1));
    }
    MatrixXf b = MatrixXf::Constant(n, 1, 1, 1.0f);
    auto LU = A.decompositionLU();
    auto x = LU.solve(b).eval();
    auto check = (A * x).eval();
    EXPECT_TRUE(MatrixNear(check, b, 1e-4));
}

TEST(LinAlgTest, LUDeterminantDynamic)
{
    float data[1][3][3] = {{
        {6, 4, 1},
        {4, 5, 1},
        {1, 1, 6}
    }};
    MatrixXfR A = MatrixXfR ::fromArray(data);
    auto LU = A.decompositionLU();
    float det = static_cast<float>(LU.determinant().eval());
    EXPECT_FLOAT_EQ(81.0f, det);
}

TEST(LinAlgTest, LUInverseDynamic)
{
    float data[1][3][3] = {{
        {6, 4, 1},
        {4, 5, 1},
        {1, 1, 6}
    }};
    MatrixXfR A = MatrixXfR::fromArray(data);
    auto LU = A.decompositionLU();
    auto inv = LU.inverse().eval();
    auto product = (A * inv).eval();
    EXPECT_TRUE(MatrixNear(product, MatrixXf::Identity(3, 3, 1), 1e-5));
}
