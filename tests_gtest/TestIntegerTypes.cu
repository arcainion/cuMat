#include "Utils.h"

using namespace cuMat;

TEST(IntegerTest, LongMatrixBasicOps)
{
    long aData[1][2][3] = {{
        {10, 20, 30},
        {40, 50, 60}
    }};
    long bData[1][2][3] = {{
        {1, 2, 3},
        {4, 5, 6}
    }};
    MatrixXlR a = MatrixXlR::fromArray(aData);
    MatrixXlR b = MatrixXlR::fromArray(bData);

    auto sum = (a + b).eval();
    long sumExpected[1][2][3] = {{
        {11, 22, 33},
        {44, 55, 66}
    }};
    EXPECT_TRUE(MatrixNear(sum.template cast<float>(), MatrixXlR::fromArray(sumExpected).template cast<float>(), 1e-6));

    auto diff = (a - b).eval();
    long diffExpected[1][2][3] = {{
        {9, 18, 27},
        {36, 45, 54}
    }};
    EXPECT_TRUE(MatrixNear(diff.template cast<float>(), MatrixXlR::fromArray(diffExpected).template cast<float>(), 1e-6));

    auto prod = a.cwiseMul(b).eval();
    long prodExpected[1][2][3] = {{
        {10, 40, 90},
        {160, 250, 360}
    }};
    EXPECT_TRUE(MatrixNear(prod.template cast<float>(), MatrixXlR::fromArray(prodExpected).template cast<float>(), 1e-6));
}

TEST(IntegerTest, LongMatrixReductions)
{
    long data[1][2][3] = {{
        {1, 2, 3},
        {4, 5, 6}
    }};
    MatrixXlR m = MatrixXlR::fromArray(data);
    long sum = static_cast<long>(m.sum());
    EXPECT_EQ(21, sum);
    long maxc = static_cast<long>(m.maxCoeff());
    EXPECT_EQ(6, maxc);
    long minc = static_cast<long>(m.minCoeff());
    EXPECT_EQ(1, minc);
    long prod = static_cast<long>(m.prod());
    EXPECT_EQ(720, prod);
}

TEST(IntegerTest, LongLongMatrixBasicOps)
{
    long long aData[1][2][2] = {{
        {100, 200},
        {300, 400}
    }};
    long long bData[1][2][2] = {{
        {5, 10},
        {15, 20}
    }};
    MatrixXllR a = MatrixXllR::fromArray(aData);
    MatrixXllR b = MatrixXllR::fromArray(bData);

    auto sum = (a + b).eval();
    long long sumExpected[1][2][2] = {{
        {105, 210},
        {315, 420}
    }};
    EXPECT_TRUE(MatrixNear(sum.template cast<float>(), MatrixXllR::fromArray(sumExpected).template cast<float>(), 1e-6));

    auto prod = a.cwiseMul(b).eval();
    long long prodExpected[1][2][2] = {{
        {500, 2000},
        {4500, 8000}
    }};
    EXPECT_TRUE(MatrixNear(prod.template cast<float>(), MatrixXllR::fromArray(prodExpected).template cast<float>(), 1e-6));
}
