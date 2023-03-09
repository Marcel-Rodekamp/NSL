add_executable(test_leapfrog test_leapfrog.cpp ../test_main.cpp)
target_link_libraries(test_leapfrog NSL Catch2::Catch2)
add_test(NAME test_leapfrog COMMAND test_leapfrog)


add_executable(test_leapfrog_convergence test_leapfrog_convergence.cpp)
target_link_libraries(test_leapfrog_convergence NSL)
