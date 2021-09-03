FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2
        GIT_TAG v2.x) # need master for benchmark::benchmark

FetchContent_MakeAvailable(
        Catch2
)

target_link_libraries(NSL_TEST INTERFACE Catch2::Catch2)
target_link_libraries(NSL_BENCHMARK INTERFACE Catch2::Catch2)
