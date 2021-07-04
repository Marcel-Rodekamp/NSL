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

# enable language support cuda
include(cmake/cuda.cmake)

# This is the shared library we can include in C++ code
add_library(NSL SHARED)
set_target_properties(NSL PROPERTIES LINKER_LANGUAGE CXX)

# include the source code
add_subdirectory(src/NSL)

# include executables
add_subdirectory(Executables)
