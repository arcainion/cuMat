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

The test suite covers: context management, matrix construction/operations, unary ops, binary ops, reductions, linear algebra (determinant, inverse, LU/Cholesky decomposition), matrix products, sparse matrices (CSR/CSC/ELLPACK), conjugate gradient solver, complex numbers, and iterators.

The old Catch-based test suite (`tests/`) and demo programs (`demos/`) were removed due to CUB compatibility issues with newer CUDA versions.

## Project Status

cuMat is a mature header-only library with broad functionality. A recent codebase review identified several areas for improvement.

### Critical Bugs Found

The following bugs were identified during a source code audit:

| Severity | File | Issue |
|----------|------|-------|
| **HIGH** | `cuMat/src/BinaryOps.h:43-44` | `ColsAtCompileTime` in `BinaryOp` traits uses `ColumnsLeft` for both branches instead of `ColumnsRight` when the right operand determines column count. This causes incorrect compile-time column dimension deduction for element-wise binary ops. |
| **HIGH** | `cuMat/src/Iterator.h:246-293` | `CountingInputIterator` ignores the `increment` parameter in all arithmetic operators (`++`, `--`, `+=`, `-=`, `+`, `-`). Only `operator*` and `operator[]` correctly use the stride. |
| **HIGH** | `cuMat/src/TransposeOp.h:147-151` | Non-const `TransposeOp::coeff()` returns a reference to the underlying matrix (stored by value), which becomes a dangling reference if the child expression is a temporary. |
| **HIGH** | `cuMat/src/ProductOp.h:214-220` | `ProductOp::batches()` checks compile-time `BatchesLeft == 1` which fails when `BatchesLeft` is `Dynamic` but the runtime value is 1, returning the wrong batch count. |

### Medium-Priority Issues

- **Context.h:371** — `createLaunchConfig1D` silently truncates `Index` (64-bit) to `unsigned int` on large matrices.
- **ReductionOps.h:331-350** — Thread reduction kernel doesn't use the initial value; undefined behavior for empty batches.
- **SparseMatrix.h:468-509** — `operator<<` for CSC and ELLPACK formats unconditionally depends on Eigen interop, even when `CUMAT_EIGEN_SUPPORT` is disabled.
- **ConjugateGradient.h:69** — CG solver statically asserts that batch count is not `Dynamic`, unnecessarily limiting use with runtime-sized batches.
- **SimpleRandom.h:237** — Random fill kernel always uses 1 block, limiting parallelism on large matrices.
- **CholeskyDecomposition.h:108** — Typo in error message: "leaading minor" → "leading minor".

See the [issue tracker](https://github.com/arcainion/cuMat/issues) for the full list.

### Test Coverage Gaps

The gtest-based test suite in `tests_gtest/` has **132 passing tests** across 11 test suites, but the following areas lack coverage:

**High priority:**
- Unary math ops: `cwiseAsin`, `cwiseAcos`, `cwiseAtan`, `cwiseSinh`, `cwiseCosh`, `cwiseTanh`, `cwiseRsqrt`, `cwiseCbrt`, `cwiseRcbrt`, `cwiseBinaryNot`, `cwiseInverseCheck`
- Compound assignment operators: `/=`, `%=`, `&=`, `|=`, matrix `*=`
- Reduction algorithm variants: `Segmented`, `Thread`, `Block<N>`, `Device<N>` (only `Warp` is tested)
- Edge cases: empty matrices (0×0), single-element matrices, very large matrices

**Medium priority:**
- Sparse matrix ops: CSC/ELLPACK SpMV, sparse matrix-matrix product, `sparseView()`, `direct()`
- Batch slicing: `slice()`, `segment()`, `head()`, `tail()`
- `asDiagonal()`, `diagonal()`, `swapAxis()`
- Custom expression operations: `unaryExpr()`, `binaryExpr()`, `NullaryExpr()`
- Eigen interop: `toEigen()`, `fromEigen()`

**Low priority:**
- `computeInverseAndDet()` for dynamic-size matrices
- Complex op gaps: `cwiseMul`, `cwiseDiv`, `cwisePow`, complex reductions
- Integer types beyond `int`
- IO / `operator<<`
- CG solver metadata and non-convergent failure path

### Missing Features (API Gaps)

- **SparseMatrix** has no direct constructor accepting raw device pointers (unlike dense `Matrix`)
- **CSC/ELLPACK** `operator<<` depends on Eigen interop (CSR has a native implementation)
- **ConjugateGradient** does not support `Dynamic` batch sizes
- No matrix-matrix element-wise `operator*` and `operator/` (use `cwiseMul()` / `cwiseDiv()` to avoid ambiguity with matrix product)
- Null-matrix specializations flagged as TODO in `Matrix.h:67`
- BLAS Level 1 operations (`axpy`, `copy`, `scal`) are available via cuBLAS but not exposed as standalone high-level ops

## License
cuMat is shipped under the permissive [MIT](https://choosealicense.com/licenses/mit/) license.

## Bug reports
If you find bugs in the library, feel free to open an issue. I will continue to use this library in future projects and therefore continue to improve and extend this library. Of course, pull requests are more than welcome.