add_executable(test_CG test_CG.cpp ../../test_main.cpp)
target_link_libraries(test_CG NSL Catch2::Catch2)
add_test(NAME test_CG COMMAND test_CG)

