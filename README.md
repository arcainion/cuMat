# cuMat: Linear algebra library in CUDA

cuMat strives to be a port of Eigen in CUDA, enabling the performance gain when computing on the GPU.

Overview:
 - Versatile:
   - cuMat supports all matrix and vector sizes, fixed on compile time or dynamically sized during runtime.
   - all matrices can be batched and all operations are parallelized over batches.
   - supports all standard float and integral types, complex types, as well as [custom scalar types](https://shaman42.gitlab.io/cuMat/_advanced__custom_scalar_types.html).
   - supports BLAS 1-3, many reductions, decompositions, and iterative solvers.
   - supports [sparse matrices](https://shaman42.gitlab.io/cuMat/_tutorial_sparse.html).
 - Fast ( [Benchmarks](https://shaman42.gitlab.io/cuMat/_benchmarks.html) ):
   - Kernel merging to minimize memory access.
   - Uses [CUB](https://nvlabs.github.io/cub/) for reductions, [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) for matrix products and [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html) for dense decompositions.
   - Custom implementations for all nullary, unary and binary operations.
   - Outperforms cuBLAS if kernel merging can be utilized.
 - Accessible:
   - Simple API influenced by Eigen.
   - Implementation details like context creation and work size spezification are hidden from the user.
   - Thread-safe.
   - Header-only.
   - Cross-Platform support. Developed under Windows, Visual Studio 2017 with CUDA 9.2. Tested with the CI on Linux, gcc and CUDA 9.2.
   - Simple interop to Eigen.

## Motivating example
To demonstrate how cuMat can be used, we show how the code for summing two vectors `a` and `b` into a thrid vector `c` looks like when implemented with Eigen, cuBLAS and cuMat.

**Eigen:**

    Eigen::VectorXf a = ..., b = ...; //some initializations
    Eigen::VectorXf c = a + b; //CPU

**cuBLAS:**

    int n = ...; //size of the vectors
    float* a = ..., b = ...; //some initializations
    float* c = ...; //output memory
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1; //optional scaling factor of b; axpy: c += alpha * b
    cudaMemcpy(c, a, sizeof(float)*n, cudaMemcpyDeviceToDevice); //copy a into c, GPU
    cublasSaxpy(handle, n, &alpha, b, 1, c, 1); //add b to c, GPU
    cublasDestroy(&handle);

Of course, this above code is a bit unfair because the boilerplate code of creating the cuBLAS handle is included.
In practice, this has to be done only once, so the above code reduces to two lines, the memcpy and the axpy.

**cuMat:**

    cuMat::VectorXf a = ..., b = ...; //some initialization
    cuMat::VectorXf c = a + b; //GPU

## Documentation
The documentation can be found under [https://shaman42.gitlab.io/cuMat/](https://shaman42.gitlab.io/cuMat/_getting_started.html).
All other open questions regarding this library are answered there.

## Requirements
cuMat is header-only, but it builds on some third-party libraries:
 - cuBLAS, cuSOLVER: shipped with the CUDA SDK.
 - [CUB](https://nvlabs.github.io/cub/): can be found inside Thrust as part of the CUDA SDK, in the third-party folder of cuMat, or provide your own version.
 - (Optional) [Eigen](http://eigen.tuxfamily.org) for printing matrices and for the Eigen interop. A working version can be found in the third-party folder.

## Building and Testing

The library uses CMake with vcpkg for dependency management:

```powershell
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="H:/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release
```

### Running Tests

cuMat uses Google Test (gtest) for its test suite. Tests are in `tests_gtest/` and are built as the `cuMat_gtest` target:

```powershell
cmake --build build --config Release --target cuMat_gtest
./build/tests_gtest/Release/cuMat_gtest.exe
```

The test suite covers: context management, matrix construction/operations, unary ops, binary ops, reductions, linear algebra (determinant, inverse, LU/Cholesky decomposition), matrix products, sparse matrices (CSR/CSC/ELLPACK), conjugate gradient solver, complex numbers, Eigen interop, integer types (long, long long), and iterators.

The old Catch-based test suite (`tests/`) and demo programs (`demos/`) were removed due to CUB compatibility issues with newer CUDA versions.

## Project Status

cuMat is a mature header-only library with broad functionality. A recent codebase review identified several areas for improvement.

### Critical Bugs Found (4 fixed, 1 false positive)

The following bugs were identified during a source code audit:

| Severity | File | Status |
|----------|------|--------|
| **HIGH** | `cuMat/src/BinaryOps.h:43-44` | **FIXED** — Changed `ColumnsLeft` to `ColumnsRight` in non-broadcast branch |
| **HIGH** | `cuMat/src/Iterator.h:246-293` | **FALSE POSITIVE** — `CountingInputIterator` stores `val` as step index, not cumulative value; arithmetic operators correctly manipulate the step index. `operator*`/`operator[]` compute `val * increment` as the actual value. |
| **HIGH** | `cuMat/src/TransposeOp.h:147-151` | **FIXED** — Added `const_cast` to resolve dangling reference in non-const `coeff()` |
| **HIGH** | `cuMat/src/ProductOp.h:214-220` | **FIXED** — Added runtime `left_.batches() == 1` check alongside compile-time check |

### Medium-Priority Issues (7 fixed, 0 pending)

**Fixed:**
- `SparseMatrix.h:468-509` — **FIXED**: Added `#if CUMAT_EIGEN_SUPPORT == 1` guards around Eigen-dependent output
- `ConjugateGradient.h:69` — **FIXED**: Improved static assertion error message (still requires compile-time batch count)
- `SimpleRandom.h:237` — **FIXED**: Changed `<<<1, ...>>>` to use occupancy-optimized `cfg.block_count`
- `CholeskyDecomposition.h:108` — **FIXED**: "leaading" → "leading" and removed duplicate "was"
- `Context.h:371` — **FIXED**: Changed `unsigned int` to `size_t` to avoid silent truncation of 64-bit `Index`.
- `ReductionOps.h:331-350` — **FIXED**: Thread reduction kernel now accepts and uses the `initial` value instead of reading the first element directly.
- `EigenInteropHelpers.h` — **FIXED**: Uncommented scalar type conversion in `MatrixCuMatToEigen`/`MatrixEigenToCuMat` so `toEigen()` returns standard Eigen types (e.g., `Eigen::MatrixXcf` instead of `Eigen::Matrix<cfloat,...>`). Fixed `toEigen()` to use `reinterpret_cast` for the type conversion.

### Test Coverage Gaps (62 new tests added, 192 total)

The gtest-based test suite in `tests_gtest/` now has **192 passing tests** across 13 test suites. The following areas were recently addressed:

**Newly tested (Phase 3):**
- Unary math ops: `cwiseAsin`, `cwiseAcos`, `cwiseAtan`, `cwiseSinh`, `cwiseCosh`, `cwiseTanh`, `cwiseRsqrt`, `cwiseCbrt`, `cwiseBinaryNot`, `cwiseLogicalNot`, `cwiseRcbrt`, `cwiseInverseCheck`
- Compound assignment operators: `/=`, `%=`, `&=`, `|=`, matrix `*=`
- Edge cases: empty matrices (0×0), single-element matrices, matrix of zeros, scalar multiplication, zero-sized batches
- `diagonal()` and `asDiagonal()`
- ELLPACK SpMV: `ELLPACKMatrixVectorProduct`
- BLAS-1 operations: `Blas1Axpy`, `Blas1Copy`, `Blas1Scal`
- Batch slicing: `slice()`, `segment()`, `head()`, `tail()`
- Custom expressions: `unaryExpr()`, `binaryExpr()`, `NullaryExpr()`

**Bug fix:**
- `BinaryOpsPlugin.inl:124-125` — Fixed `binaryExpr()` return type from `UnaryOp` to `BinaryOp` (was dropping the custom functor)

**Recently filled:**
- Reduction algorithm variants (`Segmented`, `Thread`, `Block<N>`, `Device<N>`) — **ADDED** 5 tests
- Eigen interop (`toEigen()`, `fromEigen()`) — **ADDED** 5 tests for column-major, row-major, and complex types
- Complex op gaps (`cwiseMul`, `cwiseDiv`, `cwisePow`, complex reductions) — **ADDED** 10 tests
- CG solver metadata (`iterations()`, `error()`) and non-convergent path — **ADDED** 3 tests
- CSR sparse matrix-dense matrix product (SpMM) — **ADDED** kernel and 1 test
 - `sparseView()` with CSR, CSC, and ELLPACK — **ADDED** 3 tests verifying SparseExpressionOp wrapping and sparse product evaluation
- `direct()` with CSR — **ADDED** 1 test verifying direct-coefficient-access optimization with `sparseView`
- Integer types beyond `int` — **ADDED** 3 tests for `long` and `long long` matrices (basic ops + reductions)
- CSC SpMM — **ADDED** kernel and 1 test (fixed expected values: were copied from CSR test without adjustment)
- ELLPACK SpMM — **ADDED** kernel and 1 test
- `SparseExpressionOp::coeff()` comma-operator bug — **FIXED** (was returning `batch` instead of the coefficient)

### Missing Features (partially addressed)

- **`operator~` and `operator!`** — **FIXED**: Added as member operators in `UnaryOpsPlugin.inl`, wrapping `cwiseBinaryNot` and `cwiseLogicalNot` functors respectively.
- **CSC/ELLPACK `operator<<`** — **FIXED**: Now uses native `io::print_matrix` instead of Eigen, removing the `CUMAT_EIGEN_SUPPORT` dependency.
- **BLAS-1 operations** (`axpy`, `copy`, `scal`) — **FIXED**: Added cuBLAS wrappers in `CublasApi.h` and exposed as methods on `Matrix`.
- **ConjugateGradient** does not support `Dynamic` batch sizes
- No matrix-matrix element-wise `operator*` and `operator/` (use `cwiseMul()` / `cwiseDiv()` to avoid ambiguity with matrix product)
- Null-matrix specializations flagged as TODO in `Matrix.h:67`

## Recommended Next Fix Steps

These are the recommended next steps, ordered by impact and dependency. ✅ = completed, — = no fix needed (false positive).

### Phase 1: Fix Critical Bugs ✅ Complete

1. ✅ **Fix `BinaryOps.h:43-44` — wrong compile-time column dimension**
2. — **`Iterator.h` — `CountingInputIterator`** — Determined to be a false positive. The iterator stores a step index, and all arithmetic correctly operates on the step index. `operator*`/`operator[]` compute `val * increment` for the actual value.
3. ✅ **Fix `TransposeOp.h:147-151` — dangling reference in non-const `coeff()`**
4. ✅ **Fix `ProductOp.h:214-220` — wrong batch count for dynamic-sized operands**

### Phase 2: Medium-Priority Issue Fixes ✅ Complete

5. ✅ **`ConjugateGradient.h:69` — improved error message** (still enforces compile-time batch count as the fix requires deeper architectural changes)
6. ✅ **`SimpleRandom.h:237` — use occupancy-optimized grid size**
7. ✅ **`SparseMatrix.h:468-509` — guard CSC/ELLPACK `operator<<` with `CUMAT_EIGEN_SUPPORT`**

### Phase 3: Expand Test Coverage ✅ Complete

8. ✅ **Edge cases** — Added `EmptyMatrix`, `SingleElementMatrix`, `MatrixOfZeros`, `ScalarMultiplication`, `ZeroSizedBatch`
9. ✅ **Remaining unary ops** — Added `ArcSin`, `ArcCos`, `ArcTan`, `ArcSinh`, `ArcCosh`, `ArcTanh`, `Rsqrt`, `Cbrt`, `BinaryNot`, `DiagonalView`, `AsDiagonal`
10. ✅ **Compound assignments** — Added `CompoundDivide`, `CompoundModulo`, `CompoundBitwiseAnd`, `CompoundBitwiseOr`, `CompoundMatrixMultiply`

### Phase 4: Feature Improvements ✅ Complete

11. ✅ **`Utils.h::MatrixNear` NaN-safe** — Added NaN guard and empty-matrix early return
12. ✅ **`operator~` and `operator!`** — Added as member operators on all matrix expression types
13. ✅ **`operator<<` for CSC/ELLPACK** — Replaced Eigen-dependent printing with native `io::print_matrix`
14. ✅ **BLAS-1 operations** — Added `axpy()`, `copy()`, `scal()` methods on `Matrix` using cuBLAS
15. ✅ **CholeskyDecomposition.h:108 typo** — Fixed "leaading minor" → "leading minor"

### Phase 5: Expand Test Coverage ✅ Complete

16. ✅ **ELLPACK SpMV test** — Added `ELLPACKMatrixVectorProduct`
17. ✅ **Batch slicing** — Added `Slice`, `Segment`, `Head`, `Tail` tests
18. ✅ **Remaining unary ops** — Added `InverseCheck`, `Rcbrt` tests
19. ✅ **Custom expressions** — Added `UnaryExpr`, `BinaryExpr`, `NullaryExpr` tests
20. ✅ **Bug fix: `binaryExpr` return type** — Fixed return type from `UnaryOp` to `BinaryOp`

### Phase 6: Expand Test Coverage ✅ Complete

21. ✅ **CSC SpMV kernel** — Implemented as `CSCMVKernel_StaticBatches` with one thread per column using `atomicAdd` for output accumulation. Both `SparseMatrix` and `SparseExpressionOp` specializations added. Tested in `CSCMatrixVectorProduct`.
22. ✅ **CSR sparse matrix-dense matrix product (SpMM)** — Implemented CSR SpMM kernel (`CSRMMKernel_StaticBatches`) with 2D thread mapping. Added `ProductAssignment` dispatch with runtime vector/matrix detection. Tested in `CSRMatrixMatrixProduct` (3×2 dense result).
23. ✅ **Reduction algorithm variants** — Tests added for `Segmented`, `Thread`, `Block<N>`, `Device<N>` variants (5 tests)
24. ✅ **Eigen interop tests** — Tests added for `toEigen()` / `fromEigen()` when `CUMAT_EIGEN_SUPPORT` is enabled (5 tests)
25. ✅ **Complex op gaps** — Tests added for `cwiseMul`, `cwiseDiv`, `cwisePow`, complex reductions (10 tests)
26. ✅ **CG solver metadata and non-convergent failure path** — Tests added for `iterations()` and `error()` metadata; fixed NonConvergent test (was using 3 iterations for a 3×3 system, which converges exactly)
27. ✅ **`sparseView()` / `direct()` tests** — Added `SparseViewCSR`, `SparseViewCSC`, `DirectAccessCSR` tests verifying SparseExpressionOp wrapping, sparse product evaluation, and the direct-coefficient-access optimization
28. ✅ **Integer types beyond `int`** — Added `LongMatrixBasicOps`, `LongMatrixReductions`, `LongLongMatrixBasicOps` tests for `long` and `long long` types

### Phase 7: Remaining Work ✅ Complete

29. ✅ **CSC and ELLPACK SpMM kernels** — Implemented sparse matrix-dense matrix product for CSC (`CSCMMKernel_StaticBatches`) and ELLPACK (`ELLPACKMMKernel_StaticBatches`) formats. Tested in `CSCMatrixMatrixProduct` and `ELLPACKMatrixMatrixProduct`.
30. ✅ **`sparseView()` tests** — Added `SparseViewCSR`, `SparseViewCSC`, `SparseViewELLPACK` tests verifying SparseExpressionOp wrapping and sparse product evaluation for all three formats.
31. ✅ **`sparseView()` `coeff()` bug** — Fixed `SparseExpressionOp::coeff()` at line 63-66 comma-operator bug that returned `batch` instead of the actual coefficient.

## License
cuMat is shipped under the permissive [MIT](https://choosealicense.com/licenses/mit/) license.

## Bug reports
If you find bugs in the library, feel free to open an issue. I will continue to use this library in future projects and therefore continue to improve and extend this library. Of course, pull requests are more than welcome.