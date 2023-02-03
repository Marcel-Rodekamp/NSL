FUNCTION(add_NSL_benchmark TARGET SRC_FILE)
    get_filename_component(OUT_DIR ${SRC_FILE} DIRECTORY)
    add_executable(${TARGET} ${SRC_FILE})
    set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})
    target_link_libraries(${TARGET} NSL)
ENDFUNCTION()

add_NSL_benchmark(bench_tensor_add Tensor/bench_tensor_add.cpp)
