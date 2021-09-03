# Externally provided libraries
#FetchContent_Declare(googletest
#        GIT_REPOSITORY https://github.com/google/googletest.git
#        GIT_TAG v1.10.x)

FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG master) # need master for benchmark::benchmark

FetchContent_MakeAvailable(
#        googletest
        googlebenchmark
)

set(BENCHMARK_ENABLE_TESTING DISABLE)

target_link_libraries(NSL benchmark::benchmark)
