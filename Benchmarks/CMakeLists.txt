set(BENCHMARK_TARGETS)
set(BENCHMARK_COMMANDS)

FUNCTION(add_NSL_benchmark TARGET SRC_FILE)
    get_filename_component(OUT_DIR ${SRC_FILE} DIRECTORY)
    add_executable(${TARGET} ${SRC_FILE})
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})
    target_link_libraries(${TARGET} NSL)
    set(BENCHMARK_TARGETS ${BENCHMARK_TARGETS} ${TARGET} PARENT_SCOPE)
    set(BENCHMARK_COMMANDS ${BENCHMARK_COMMANDS} COMMAND $<TARGET_FILE:${TARGET}> PARENT_SCOPE)
ENDFUNCTION()

add_NSL_benchmark(bench_tensor_add Tensor/bench_tensor_add.cpp)

# Meta target to run all benchmarks
add_custom_target(benchmark
    ${BENCHMARK_COMMANDS}
    DEPENDS ${BENCHMARK_TARGETS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running all benchmarks in ${CMAKE_CURRENT_SOURCE_DIR}"
)
set_target_properties(benchmark PROPERTIES EXCLUDE_FROM_ALL TRUE)