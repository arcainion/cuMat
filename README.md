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
 - [CUB](https://nvlabs.github.io/cub/): included in the CUDA Toolkit (via Thrust).
 - [Eigen](http://eigen.tuxfamily.org): used by tests for Eigen interop support. Managed via vcpkg.

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

The test suite covers: context management, matrix construction/operations, unary ops, binary ops, reductions, linear algebra (determinant, inverse, LU/Cholesky decomposition), matrix products, sparse matrices (CSR/CSC/ELLPACK) including batched cwise ops, conjugate gradient solver, complex numbers, Eigen interop, integer types (long, long long), and iterators.

The old Catch-based test suite (`tests/`) and demo programs (`demos/`) were removed due to CUB compatibility issues with newer CUDA versions.

## Project Status

cuMat is a mature header-only library with broad functionality. See [CHANGE_LIST.md](CHANGE_LIST.md) for a complete record of all fixes, improvements, and changes.

### Test Coverage
The gtest-based test suite has **205 passing tests** across 14 test suites covering: context management, matrix construction/operations, unary ops, binary ops, reductions, linear algebra (determinant, inverse, LU/Cholesky decomposition), matrix products, sparse matrices (CSR/CSC/ELLPACK) including batched cwise ops, conjugate gradient solver, complex numbers, Eigen interop, integer types (long, long long), iterators, and fromLinear optimization verification.

### Known Limitations
- ConjugateGradient does not support `Dynamic` batch sizes
- No matrix-matrix element-wise `operator*` and `operator/` (use `cwiseMul()` / `cwiseDiv()` to avoid ambiguity with matrix product)
- Null-matrix specializations flagged as TODO in `Matrix.h:67`

## Completed Work

All planned phases of fixes, improvements, test expansions, performance optimizations, and dependency migration are now complete. See [CHANGE_LIST.md](CHANGE_LIST.md) for the full detailed record.

## Remaining Optimization Opportunities

Several performance items from the original audit have been addressed (see [CHANGE_LIST.md](CHANGE_LIST.md)). The following opportunities remain:

### High Priority

- **Missing `const __restrict__` on all kernel parameters** — Blocked: MSVC host compiler rejects `__restrict__` on value-type kernel parameters. Only raw pointer parameters are annotated.

### Medium Priority

- **Thread reduction kernel selection thresholds** — Thresholds in `ReductionAlgorithmSelection.h` were tuned on an RTX 2070 and may be suboptimal on other architectures.
- ~~**`StridedMatrixInputIterator` — 3 divisions + 3 modulos per element** — `Iterator.h:64-78` — **DONE**: Replaced with sequential stride decomposition; falls back to original formula when strides are equal.~~
- ~~**Synchronization-heavy debug mode** — `CUMAT_VERBOSE_ERROR_CHECKING=1` — **DONE**: Replaced `cudaDeviceSynchronize()` with `cudaStreamSynchronize(stream_)` in cuBLAS/cuSOLVER wrappers; removed sync entirely from `cudaSafeCall` (CUDA runtime APIs return errors synchronously).~~

### Low Priority

- ~~**`typeid()` in debug logging forces RTTI emission** — 17 `CUMAT_LOG_DEBUG` calls use `typeid(T).name()`, forcing type_info emission for every template instantiation. **DONE**: Added `internal::type_name<T>()` helper based on `__FUNCSIG__`/`__PRETTY_FUNCTION__` (no RTTI). Replaced all 15 call sites and removed `#include <typeinfo>` from `Context.h`.~~
- **`createLaunchConfig2D/3D` use 1D thread blocks** — Always creates `dim3(bestBlockSize, 1, 1)` blocks; no 2D block-level spatial locality is exploited.
- **Hardcoded warp size of 32** — `ReductionOps.h:360-387` hardcodes `32` in the warp reduction kernel.
- **Dead code** — `Context.h:374-382` has a commented-out hardcoded block size of 256.

## License
cuMat is shipped under the permissive [MIT](https://choosealicense.com/licenses/mit/) license.

## Bug reports
If you find bugs in the library, feel free to open an issue. I will continue to use this library in future projects and therefore continue to improve and extend this library. Of course, pull requests are more than welcome.