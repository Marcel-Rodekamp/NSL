# TODO: add matvec test.
#add_executable(test_matvec test_matvec.cpp)
#target_link_libraries(test_matvec NSL_TEST NSL)

add_executable(test_adjoint test_adjoint.cpp ../test_main.cpp)
target_link_libraries(test_adjoint NSL Catch2::Catch2)
add_test(NAME test_adjoint COMMAND test_adjoint)

add_executable(test_conj test_conj.cpp ../test_main.cpp)
target_link_libraries(test_conj NSL Catch2::Catch2)
add_test(NAME test_conj COMMAND test_conj)

add_executable(test_mat_exp test_mat_exp.cpp ../test_main.cpp)
target_link_libraries(test_mat_exp NSL Catch2::Catch2)
add_test(NAME test_mat_exp COMMAND test_mat_exp)

add_executable(test_transpose test_transpose.cpp ../test_main.cpp)
target_link_libraries(test_transpose NSL Catch2::Catch2)
add_test(NAME test_transpose COMMAND test_transpose)
