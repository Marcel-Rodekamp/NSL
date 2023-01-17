add_executable(test_IO_h5io test_IO_h5io.cpp ../test_main.cpp)
target_link_libraries(test_IO_h5io NSL Catch2::Catch2)
add_test(NAME test_IO_h5io COMMAND test_IO_h5io)



