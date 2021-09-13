# check if torch can be found on the system default
find_package(Torch QUIET)

# if not found download a version and populate it
if(Torch_FOUND)
    message(STATUS "Found Installed Torch: ${TORCH_INCLUDE_DIRS}")
else()
    if(APPLE)
        message(STATUS "Getting Torch: Apple CPU")
        FetchContent_Declare(
            Torch
            URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip
        )
    else()
        message(STATUS "Getting Torch: Linux CPU")
        FetchContent_Declare(
            Torch
            URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
        )
    endif()
    
    FetchContent_MakeAvailable(Torch)
    
    set(Torch_DIR "${FETCHCONTENT_BASE_DIR}/torch-src/share/cmake/Torch")
    
    find_package(Torch REQUIRED)
endif()

message(STATUS "Torch dir: ${Torch_DIR}")
message(STATUS "Adding TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
message(STATUS "Adding TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(NSL
    ${TORCH_LIBRARIES}
)
