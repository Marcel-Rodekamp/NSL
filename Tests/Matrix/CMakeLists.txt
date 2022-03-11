add_executable(test_identity test_identity.cpp ../test_main.cpp)
target_link_libraries(test_identity NSL Catch2::Catch2)
add_test(NAME test_identity COMMAND test_identity)
