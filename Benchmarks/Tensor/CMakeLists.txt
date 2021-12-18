add_executable(benchmark_tensor benchmark_tensor.cpp ../benchmark_main.cpp)
target_link_libraries(benchmark_tensor NSL Catch2::Catch2)
