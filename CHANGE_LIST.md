# Change List

All completed changes, fixes, and improvements to cuMat.

## Critical Bugs Fixed

| Severity | File | Fix |
|----------|------|-----|
| HIGH | `cuMat/src/BinaryOps.h:43-44` | Changed `ColumnsLeft` to `ColumnsRight` in non-broadcast branch |
| HIGH | `cuMat/src/TransposeOp.h:147-151` | Added `const_cast` to resolve dangling reference in non-const `coeff()` |
| HIGH | `cuMat/src/ProductOp.h:214-220` | Added runtime `left_.batches() == 1` check alongside compile-time check |

## Medium-Priority Issues Fixed

- `SparseMatrix.h:468-509` — Added `#if CUMAT_EIGEN_SUPPORT == 1` guards around Eigen-dependent output
- `ConjugateGradient.h:69` — Improved static assertion error message
- `SimpleRandom.h:237` — Changed `<<<1, ...>>>` to use occupancy-optimized `cfg.block_count`
- `CholeskyDecomposition.h:108` — Fixed "leaading" → "leading" typo and removed duplicate "was"
- `Context.h:371` — Changed `unsigned int` to `size_t` to avoid silent truncation of 64-bit `Index`
- `ReductionOps.h:331-350` — Thread reduction kernel now accepts and uses the `initial` value instead of reading the first element directly
- `EigenInteropHelpers.h` — Uncommented scalar type conversion so `toEigen()` returns standard Eigen types; Fixed `toEigen()` to use `reinterpret_cast` for type conversion

## Feature Improvements

- `Utils.h::MatrixNear` NaN-safe — Added NaN guard and empty-matrix early return
- `operator~` and `operator!` — Added as member operators on all matrix expression types
- `operator<<` for CSC/ELLPACK — Replaced Eigen-dependent printing with native `io::print_matrix`
- BLAS-1 operations — Added `axpy()`, `copy()`, `scal()` methods on `Matrix` using cuBLAS
- CSC/ELLPACK `operator<<` — Uses native `io::print_matrix` instead of Eigen, removing `CUMAT_EIGEN_SUPPORT` dependency
- CSC SpMV kernel — Implemented `CSCMVKernel_StaticBatches` with one thread per column using `atomicAdd`
- CSR SpMM kernel — Implemented `CSRMMKernel_StaticBatches` with 2D thread mapping; runtime vector/matrix detection in `ProductAssignment`
- CSC SpMM kernel — Implemented `CSCMMKernel_StaticBatches` for sparse matrix-dense matrix product
- ELLPACK SpMM kernel — Implemented `ELLPACKMMKernel_StaticBatches` for sparse matrix-dense matrix product
- `BinaryOpsPlugin.inl:124-125` — Fixed `binaryExpr()` return type from `UnaryOp` to `BinaryOp`
- `SparseExpressionOp::coeff()` — Fixed comma-operator bug that returned `batch` instead of the actual coefficient

## Test Coverage Additions (203 tests, 13 suites)

### Phase 3 Tests
- Unary math ops: `cwiseAsin`, `cwiseAcos`, `cwiseAtan`, `cwiseSinh`, `cwiseCosh`, `cwiseTanh`, `cwiseRsqrt`, `cwiseCbrt`, `cwiseBinaryNot`, `cwiseLogicalNot`, `cwiseRcbrt`, `cwiseInverseCheck`
- Compound assignment operators: `/=`, `%=`, `&=`, `|=`, matrix `*=`
- Edge cases: empty matrices (0×0), single-element matrices, matrix of zeros, scalar multiplication, zero-sized batches
- `diagonal()` and `asDiagonal()`
- ELLPACK SpMV: `ELLPACKMatrixVectorProduct`
- BLAS-1 operations: `Blas1Axpy`, `Blas1Copy`, `Blas1Scal`
- Batch slicing: `slice()`, `segment()`, `head()`, `tail()`
- Custom expressions: `unaryExpr()`, `binaryExpr()`, `NullaryExpr()`

### Phase 5 Tests
- ELLPACK SpMV test (`ELLPACKMatrixVectorProduct`)
- Batch slicing (`Slice`, `Segment`, `Head`, `Tail`)
- Remaining unary ops (`InverseCheck`, `Rcbrt`)
- Custom expressions (`UnaryExpr`, `BinaryExpr`, `NullaryExpr`)

### Phase 6 Tests
- Reduction algorithm variants (`Segmented`, `Thread`, `Block<N>`, `Device<N>`) — 5 tests
- Eigen interop (`toEigen()`, `fromEigen()`) — 5 tests for column-major, row-major, and complex types
- Complex op gaps (`cwiseMul`, `cwiseDiv`, `cwisePow`, complex reductions) — 10 tests
- CG solver metadata (`iterations()`, `error()`) and non-convergent path — 3 tests
- CSR SpMM — kernel and 1 test
- `sparseView()` with CSR, CSC, ELLPACK — 3 tests
- `direct()` with CSR — 1 test
- Integer types beyond `int` — 3 tests for `long` and `long long`
- CSC SpMM — kernel and 1 test
- ELLPACK SpMM — kernel and 1 test

### Phase 7 Tests
- `sparseView()` tests for CSR, CSC, ELLPACK formats
- Batched transpose — 1 test verifying 2×2×3 → 2×3×2 transposition

## Performance Optimizations (Phase 8)

- **Thread reduction kernel** — Replaced stride-N global access with warp-cooperative block-level reduction using shared memory
- **`CUMAT_STRONG_INLINE` → `__forceinline__`** — Updated `Macros.h:201` to force inlining on hot paths
- **`createLaunchConfig1D` grid capping** — Changed from `min()` to `max()` so large workloads use 1-2 passes; applied to 2D/3D variants
- **`__launch_bounds__` annotations** — Added to every custom kernel for register allocation hints
- **Merged `ProductAssignment` specializations** — Collapsed 3 near-identical CSR/CSC/ELLPACK structs into one template with tag dispatch
- **Collapsed `DenseStorage` specializations** — Replaced 8 partial specializations with a single template, eliminating ~280 lines
- **Sparse index array caching** — Implemented column-fast 2D mapping in CSR SpMM for coalesced dense access
- **Dedicated transfer stream** — Added `transferStream_` per context for overlapping host-device transfers with kernel execution
- **Eliminated linear→coord→linear round-trip** — Added `CwiseEvalHelper` with `DirectSrc` path using `rawCoeff(index)` for direct-access sources
- **Binary search in sparse evaluator** — Replaced linear search with binary search in CSR/CSC `coordsToLinear`
- **Async `copyFromHost`/`copyToHost`** — Added async variants alongside sync versions
- **Reduced CSC atomicAdd contention** — Replaced direct global-memory `atomicAdd` in `CSCMVKernel_StaticBatches` and `CSCMMKernel_StaticBatches` with a shared-memory hash table (linear probing, 1024 slots). Each thread block accumulates contributions in shared memory first, then flushes with at most 1024 global atomicAdds per block instead of one per non-zero. Fallback to direct atomicAdd when the hash table is full.
- **Batch-inner runtime heuristic for sparse cwise evaluation** — `SparseEvaluation.h`: Replaced compile-time `#if CUMAT_SPARSE_EVAL_BATCH_INNER` with a runtime heuristic (`CUMAT_SPARSE_EVAL_BATCH_THRESHOLD`, default 4). When `batches > threshold`, the kernel switches from a 2D (outer, batch) parallel strategy to a batch-inner strategy where one thread per outer index iterates all batches, amortizing JA/IA sparsity-pattern loads. Both code paths are compiled into each kernel and selected via a `bool useBatchInner` parameter, with appropriate 1D/2D launch configs chosen at the call site. Applies to CSR, CSC, and ELLPACK formats.
- **One-thread-per-row runtime heuristic for SpMM** — `SparseProductEvaluation.h`: Added `CUMAT_SPARSE_MM_ACCUM_THRESHOLD` (default 32) for CSRMM and ELLPACKMM kernels. When `cols × Batches <= threshold`, a single thread per row iterates over all output columns and batches, amortizing JA/IA/index loads across columns. The `bool useOneThreadPerRow` runtime parameter selects between the new 1D (one-thread-per-row) and the existing 2D (thread-per-output-element) strategy at the call site based on the threshold.
- **`StridedMatrixInputIterator::fromLinear` stride-decomposition optimization** — `Iterator.h`: Replaced 3 divisions + 3 modulos per element with sequential decomposition that peels the largest stride first, reducing to 2 divisions (one for the largest stride, one for the next), 2 multiplications, and 2 subtractions. Falls back to the original independent-dimension formula (`(linear/s_i)%d_i`) when any two strides are equal, since the sequential decomposition is ambiguous in that case. This fixes a bug where equal strides (e.g., simplified Row|Batch axis stride `(1,2,2)` for a 2×3×1 matrix) caused incorrect coordinate assignment, producing reduction outputs of 0 for batches beyond the first.
- **Replaced `typeid()` calls with RTTI-free `type_name<T>()` helper** — `Logging.h`: Added `internal::type_name<T>()` returning `__FUNCSIG__` (MSVC) or `__PRETTY_FUNCTION__` (GCC/Clang/NVCC) instead of `typeid(T).name()`, avoiding forced `type_info` emission for every template instantiation. Replaced all 15 `typeid(...).name()` call sites across `Context.h` (3), `CwiseOp.h` (1), `ReductionOps.h` (3), `SparseEvaluation.h` (1), and `SparseProductEvaluation.h` (7). Removed `#include <typeinfo>` from `Context.h`.
- **Lighter-weight verbose error checking** — `Errors.h`, `CublasApi.h`, `CusolverApi.h`: Replaced `cudaDeviceSynchronize()` (syncs ALL streams/devices) with per-stream `cudaStreamSynchronize(stream_)` in cuBLAS/cuSOLVER wrappers. Removed device sync entirely from `cudaSafeCall()` (CUDA runtime APIs like `cudaMalloc`, `cudaMemcpy` return errors synchronously — the sync added no value). `cudaCheckError()` also lost its sync (no `Context` access from `Errors.h`). Debug builds with `CUMAT_VERBOSE_ERROR_CHECKING=1` are now significantly less disruptive.

### Phase 8 Tests
- Batched sparse cwise evaluation — 11 tests covering CSR, CSC, ELLPACK with copy assign, cwise negate, compound add, scalar multiply, many batches (10), and single batch

## Dependency Migration (Phase 9)

- **Replaced bundled Eigen3** — Removed `third-party/Eigen/`, uses `find_package(Eigen3 CONFIG REQUIRED)` from vcpkg; added `--threads 0` nvcc flag
- **Removed bundled Catch2** — All tests use Google Test (gtest from vcpkg); removed `third-party/catch/catch.hpp`
- **Removed bundled CUB** — CUB sourced from CUDA Toolkit 12.4 (via Thrust); removed `third-party/cub/`
- **Updated build documentation** — All references to `third-party/` removed from requirements and README
