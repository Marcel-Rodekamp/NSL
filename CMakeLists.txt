cmake_minimum_required(VERSION 3.18)


# Project name
project(NanoSystemLibrary)


# Lets support some IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


# To download required cmake modules
include(FetchContent)

# At current version of CUDA and Torch the population of the C++20 standard 
# results in cmake errors. Moving the find_package call before seeting the 
# standard is a hack which circumvents the problem.
find_package(Torch REQUIRED)

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


# add module path; libraries should be
if(APPLE)
  # .dylib, on mac
  set(CMAKE_INSTALL_RPATH @loader_path)
else()
  # .so, generically or
  set(CMAKE_INSTALL_RPATH $ORIGIN)
endif()


# include requirements on which the library is build
include(cmake/ARM64.cmake)
#include(cmake/gpu.cmake)
include(cmake/torch.cmake)
include(cmake/catch2.cmake)
include(cmake/highfive.cmake)
include(cmake/spdlog.cmake)
include(cmake/yamlcpp.cmake)

add_subdirectory(src/NSL)


add_subdirectory(Executables)


add_subdirectory(Tests)


add_subdirectory(Benchmarks)


add_subdirectory(Documentation EXCLUDE_FROM_ALL)

