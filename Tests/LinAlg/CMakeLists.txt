# TODO: add matvec test.
#add_executable(test_LinAlg_matvec test_LinAlg_matvec.cpp)
#target_link_libraries(test_LinAlg_matvec NSL_TEST NSL)

add_executable(test_LinAlg_adjoint test_LinAlg_adjoint.cpp ../test_main.cpp)
target_link_libraries(test_LinAlg_adjoint NSL Catch2::Catch2)
add_test(NAME test_LinAlg_adjoint COMMAND test_LinAlg_adjoint)

add_executable(test_LinAlg_conj test_LinAlg_conj.cpp ../test_main.cpp)
target_link_libraries(test_LinAlg_conj NSL Catch2::Catch2)
add_test(NAME test_LinAlg_conj COMMAND test_LinAlg_conj)

add_executable(test_LinAlg_mat_exp test_LinAlg_mat_exp.cpp ../test_main.cpp)
target_link_libraries(test_LinAlg_mat_exp NSL Catch2::Catch2)
add_test(NAME test_LinAlg_mat_exp COMMAND test_LinAlg_mat_exp)

add_executable(test_LinAlg_transpose test_LinAlg_transpose.cpp ../test_main.cpp)
target_link_libraries(test_LinAlg_transpose NSL Catch2::Catch2)
add_test(NAME test_LinAlg_transpose COMMAND test_LinAlg_transpose)

add_executable(test_LinAlg_shift test_LinAlg_shift.cpp ../test_main.cpp)
target_link_libraries(test_LinAlg_shift NSL Catch2::Catch2)
add_test(NAME test_LinAlg_shift COMMAND test_LinAlg_shift)

add_executable(test_LinAlg_matvec test_LinAlg_matvec.cpp ../test_main.cpp)
target_link_libraries(test_LinAlg_matvec NSL Catch2::Catch2)
add_test(NAME test_LinAlg_matvec COMMAND test_LinAlg_matvec)

add_executable(test_LinAlg_mat_mul test_LinAlg_mat_mul.cpp ../test_main.cpp)
target_link_libraries(test_LinAlg_mat_mul NSL Catch2::Catch2)
add_test(NAME test_mat_mul COMMAND test_LinAlg_mat_mul)

add_executable(test_LinAlg_det test_LinAlg_det.cpp ../test_main.cpp)
target_link_libraries(test_LinAlg_det NSL Catch2::Catch2)
add_test(NAME test_LinAlg_det COMMAND test_LinAlg_det)

add_subdirectory(Solver/)

