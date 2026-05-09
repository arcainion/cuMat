# gtest-based comprehensive test suite for cuMat
cmake_minimum_required(VERSION 3.10)

# Find GTest from vcpkg
find_package(GTest CONFIG REQUIRED)

# Enable testing
enable_testing()

# Collect all test source files
set(GTEST_TEST_SOURCES
    TestContext.cu
    TestMatrix.cu
    TestReductions.cu
    TestUnaryOps.cu
    TestBinaryOps.cu
    TestLinAlg.cu
    TestProduct.cu
    TestSparse.cu
    TestConjugateGradient.cu
    TestComplexOps.cu
    TestEigenInterop.cu
    TestIntegerTypes.cu
    TestIterators.cu
    TestFromLinear.cu
    main.cu
)

set(GTEST_TEST_HEADERS
    Utils.h
)

# Common CUDA compiler flags
set(CUDA_NVCC_FLAGS_DEBUG "-DCUMAT_CONTEXT_DEBUG_MEMORY=1 -DCUMAT_EIGEN_SUPPORT=1 -DCUMAT_PROFILING=1")

# GTest requires C++17, and the main project defaults to C++11.
# Override here: set device-side C++17, and pass /std:c++17 to the MSVC host compiler
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -Xcompiler \"/std:c++17\"")

cuda_add_executable(cuMat_gtest ${GTEST_TEST_HEADERS} ${GTEST_TEST_SOURCES})
cuda_add_cublas_to_target(cuMat_gtest)
set_target_properties(cuMat_gtest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cuMat_gtest
    ${CUDA_LIBRARIES}
    ${CUDA_cusolver_LIBRARY}
    GTest::gtest
    GTest::gtest_main
)

# Disable faulty CUB compatibility issues with CUDA 12.4 by defining compatibility macros
target_compile_definitions(cuMat_gtest PRIVATE
    CUMAT_EIGEN_SUPPORT=1
    CUMAT_PROFILING=1
    CUMAT_CONTEXT_USE_CUB_ALLOCATOR=0
    CUB_IGNORE_DEPRECATED_API
    CUMAT_CONTEXT_DEBUG_MEMORY=0
)

add_test(NAME cuMatGtest COMMAND cuMat_gtest)
