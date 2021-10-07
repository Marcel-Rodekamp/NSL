cmake_minimum_required(VERSION 3.18)


# Project name
project(NanoSystemLibrary)


# Lets support some IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


# To download required cmake modules
include(FetchContent)


# set come C++ variables.
# ToDo: Do this dynamically
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -Wall -pedantic ${CMAKE_CXX_FLAGS_DEBUG_INIT}")
# Pytorch uses long int instead of std:size_t (aka unsigned long int)
# to avoid warning we put -Wno-narrowing
# ToDo: There should be a better way to circumvent this problem!!!
set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -Wno-narrowing" )
set( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fPIC" )


# Create main library to include in C++
add_library(NSL SHARED src/NSL)
set_target_properties(NSL PROPERTIES LINKER_LANGUAGE CXX)


# Create testing library depending on Catch2
add_library(NSL_TEST SHARED Tests/test_main.cpp)
set_target_properties(NSL_TEST PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(NSL_TEST NSL)


# Create benchmarking library depending on Catch2
add_library(NSL_BENCHMARK SHARED Benchmarks/benchmark_main.cpp)
set_target_properties(NSL_BENCHMARK PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(NSL_BENCHMARK NSL)

# add module path; libraries should be
if(NOT APPLE)
  # .so, generically or
  set(CMAKE_INSTALL_RPATH $ORIGIN)
else()
  # .dylib, on mac
  set(CMAKE_INSTALL_RPATH @loader_path)
endif()

# include requirements on which the library is build
include(cmake/torch.cmake)
include(cmake/catch2.cmake)


add_subdirectory(src/NSL)


add_subdirectory(Executables)


add_subdirectory(Tests)


add_subdirectory(Benchmarks)


add_subdirectory(Documentation)
