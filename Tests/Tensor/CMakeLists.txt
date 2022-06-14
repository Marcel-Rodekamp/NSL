add_executable(test_Tensor_constructor test_Tensor_constructor.cpp ../test_main.cpp)
target_link_libraries(test_Tensor_constructor NSL Catch2::Catch2)
add_test(NAME test_Tensor_constructor COMMAND test_Tensor_constructor)

add_executable(test_Tensor_torchConversion test_Tensor_torchConversion.cpp ../test_main.cpp)
target_link_libraries(test_Tensor_torchConversion NSL Catch2::Catch2)
add_test(NAME test_Tensor_torchConversion COMMAND test_Tensor_torchConversion)

add_executable(test_Tensor_randomAccess test_Tensor_randomAccess.cpp ../test_main.cpp)
target_link_libraries(test_Tensor_randomAccess NSL Catch2::Catch2)
add_test(NAME test_Tensor_randomAccess COMMAND test_Tensor_randomAccess)

add_executable(test_Tensor_operatorAddition test_Tensor_operatorAddition.cpp ../test_main.cpp)
target_link_libraries(test_Tensor_operatorAddition NSL Catch2::Catch2)
add_test(NAME test_Tensor_operatorAddition COMMAND test_Tensor_operatorAddition)

add_executable(test_Tensor_operatorMultiplication test_Tensor_operatorMultiplication.cpp ../test_main.cpp)
target_link_libraries(test_Tensor_operatorMultiplication NSL Catch2::Catch2)
add_test(NAME test_Tensor_operatorMultiplication COMMAND test_Tensor_operatorMultiplication)

add_executable(test_Tensor_operatorCompare test_Tensor_operatorCompare.cpp ../test_main.cpp)
target_link_libraries(test_Tensor_operatorCompare NSL Catch2::Catch2)
add_test(NAME test_Tensor_operatorCompare COMMAND test_Tensor_operatorCompare)

add_executable(test_Tensor_operatorAssignment test_Tensor_operatorAssignment.cpp ../test_main.cpp)
target_link_libraries(test_Tensor_operatorAssignment NSL Catch2::Catch2)
add_test(NAME test_Tensor_operatorAssignment COMMAND test_Tensor_operatorAssignment)

add_executable(test_Tensor_conj test_Tensor_conj.cpp ../test_main.cpp)
target_link_libraries(test_Tensor_conj NSL Catch2::Catch2)
add_test(NAME test_Tensor_conj COMMAND test_Tensor_conj)

add_executable(test_Tensor_stats test_Tensor_stats.cpp ../test_main.cpp)
target_link_libraries(test_Tensor_stats NSL Catch2::Catch2)
add_test(NAME test_Tensor_stats COMMAND test_Tensor_stats)

add_executable(test_Tensor_slice test_Tensor_slice.cpp ../test_main.cpp)
target_link_libraries(test_Tensor_slice NSL Catch2::Catch2)
add_test(NAME test_Tensor_slice COMMAND test_Tensor_slice)
