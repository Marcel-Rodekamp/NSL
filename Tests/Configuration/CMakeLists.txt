add_executable(test_configuration test_configuration.cpp ../test_main.cpp)
target_link_libraries(test_configuration NSL Catch2::Catch2)
add_test(NAME test_configuration COMMAND test_configuration)
