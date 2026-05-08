#ifndef CUMAT_GTEST_UTILS_H
#define CUMAT_GTEST_UTILS_H

#include <cuMat/Core>
#include <cuMat/Dense>
#include <cuMat/Sparse>
#include <cuMat/IterativeLinearSolvers>
#include <gtest/gtest.h>

template<typename T1, typename T2>
::testing::AssertionResult MatrixNear(const cuMat::MatrixBase<T1>& actual,
    const cuMat::MatrixBase<T2>& expected, double tolerance = 1e-5)
{
    auto a = actual.eval();
    auto e = expected.eval();
    if (a.rows() != e.rows() || a.cols() != e.cols() || a.batches() != e.batches())
        return ::testing::AssertionFailure() << "Dimension mismatch";
    auto diff = (a - e).cwiseAbs().eval();
    auto maxReduction = diff.maxCoeff();
    typename decltype(maxReduction)::Scalar maxDiff(maxReduction);
    if (maxDiff > tolerance)
        return ::testing::AssertionFailure() << "Max diff: " << maxDiff;
    return ::testing::AssertionSuccess();
}

#endif
