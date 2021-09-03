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
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -Wall -pedantic ${CMAKE_CXX_FLAGS_DEBUG_INIT}")

set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC" )
set( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fPIC" )
# This is the shared library we can include in C++ code
add_library(NSL SHARED src/NSL)
set_target_properties(NSL PROPERTIES LINKER_LANGUAGE CXX)

add_library(NSL_TEST INTERFACE Tests/test_main.cpp)
set_target_properties(NSL_TEST PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(NSL_TEST INTERFACE NSL)

add_library(NSL_BENCHMARK INTERFACE Benchmarks/benchmark_main.cpp)
set_target_properties(NSL_BENCHMARK PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(NSL_BENCHMARK INTERFACE NSL)

# enable language support cuda
#include(cmake/cuda.cmake)
include(cmake/torch.cmake)
include(cmake/catch2.cmake)


# include the source code
add_subdirectory(src/NSL)

# include executables
add_subdirectory(Executables)

# include Tests
add_subdirectory(Tests)

# include Benchmarks
add_subdirectory(Benchmarks)

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

target_link_libraries(NSL benchmark::benchmark)


