cmake_minimum_required(VERSION 3.0.0)
project(cuMat)
# Main CMakeLists of cuMat

# Set prefix path for vcpkg packages
set(CMAKE_PREFIX_PATH "H:/vcpkg/installed/x64-windows" ${CMAKE_PREFIX_PATH})

# Cuda is always required
find_package(CUDA REQUIRED)
if(COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
	CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
	LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
endif()
# But Eigen does not work with CUDA, disable it globally
add_definitions(-DEIGEN_NO_CUDA=1)

# Add OpenMP if available. This is needed by cuSolver for optimal performance
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Eigen is not required by default,
# but the tests depend on it -> use bundled Eigen and Catch for tests

# Hack: The FindCUDA that is used by the CI does not correctly support target_include_directories,
#  it does not forward the include paths.
#  Hence I have to include the paths globally.
# Include third-party for Eigen and catch, but we need to handle CUB conflict
include_directories(${CMAKE_SOURCE_DIR} ${CUDA_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/third-party")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
LIST(APPEND CUDA_NVCC_FLAGS "-std=c++11")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(third-party)
add_subdirectory(cuMat)
add_subdirectory(tests_gtest)
# Demos and benchmarks disabled due to CUB compatibility issues with CUDA 12.4
# add_subdirectory(demos)
# add_subdirectory(docs)
# add_subdirectory(benchmarks)

