add_executable(example_tensor
        example_tensor.cpp)

add_executable(example_benchmark
        benchmark.cpp)

target_link_libraries(example_tensor
    NSL
)
target_link_libraries(example_benchmark
        NSL
        )

