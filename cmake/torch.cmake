FetchContent_Declare(
    Torch
    URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
)

FetchContent_MakeAvailable(Torch)

find_package(Torch REQUIRED)

message(STATUS "ADDING TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
message(STATUS "ADDING TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(NSL
    ${TORCH_LIBRARIES}
)