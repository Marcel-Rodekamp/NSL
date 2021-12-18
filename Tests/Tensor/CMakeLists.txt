add_executable(test_tensor test_tensor.cpp ../test_main.cpp)
target_link_libraries(test_tensor NSL Catch2::Catch2)

add_executable(test_timetensor test_timetensor.cpp ../test_main.cpp)
target_link_libraries(test_timetensor NSL Catch2::Catch2)

add_executable(test_compilation test_compilation.cpp ../test_main.cpp)
target_link_libraries(test_compilation NSL Catch2::Catch2)
