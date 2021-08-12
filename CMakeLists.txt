cmake_minimum_required(VERSION 3.19)

# Project name
project(NanoSystemLibrary)

# Lets support some IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# To download required cmake modules
include(FetchContent)


# set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# This is the shared library we can include in C++ code
add_library(NSL SHARED src/NSL)
set_target_properties(NSL PROPERTIES LINKER_LANGUAGE CXX)

# enable language support cuda
#include(cmake/cuda.cmake)
include(cmake/torch.cmake)


# include the source code
add_subdirectory(src/NSL)

# include executables
add_subdirectory(Executables)

# Externally provided libraries
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.10.x)

FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG master) # need master for benchmark::benchmark

FetchContent_MakeAvailable(
        googletest
        googlebenchmark)


set(BENCHMARK_FORMAT "csv")
set(BENCHMARK_OUT "filename.csv")

target_link_libraries(NSL benchmark::benchmark)


