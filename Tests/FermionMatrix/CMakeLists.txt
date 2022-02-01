add_executable(test_logDetM test_logDetM.cpp ../test_main.cpp)
target_link_libraries(test_logDetM NSL Catch2::Catch2)
add_test(NAME test_logDetM COMMAND test_logDetM)