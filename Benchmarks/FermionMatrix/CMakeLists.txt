#add_executable(bench_M bench_M.cpp)
#target_link_libraries(bench_M NSL)

#add_executable(bench_logDetM bench_logDetM.cpp)
#target_link_libraries(bench_logDetM NSL)

#add_executable(bench_MdaggerM bench_MdaggerM.cpp)
#target_link_libraries(bench_MdaggerM NSL)

add_executable(bench_exp_ring bench_exp_ring.cpp)
target_link_libraries(bench_exp_ring NSL)

add_executable(bench_diag_ring bench_diag_ring.cpp)
target_link_libraries(bench_diag_ring NSL)

add_executable(bench_exp_square bench_exp_square.cpp)
target_link_libraries(bench_exp_square NSL)

add_executable(bench_diag_square bench_diag_square.cpp)
target_link_libraries(bench_diag_square NSL)

add_executable(bench_exp_cube bench_exp_cube.cpp)
target_link_libraries(bench_exp_cube NSL)

add_executable(bench_diag_cube bench_diag_cube.cpp)
target_link_libraries(bench_diag_cube NSL)