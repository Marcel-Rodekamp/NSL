FetchContent_Declare(
        nvtx
        GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
        GIT_TAG        v3.1.0
        SOURCE_SUBDIR  c
)

FetchContent_MakeAvailable(nvtx)
target_link_libraries(NSL nvtx3-cpp)