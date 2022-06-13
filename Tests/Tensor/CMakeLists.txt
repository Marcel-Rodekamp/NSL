add_executable(test_constructor test_constructor.cpp ../test_main.cpp)
target_link_libraries(test_constructor NSL Catch2::Catch2)
add_test(NAME test_constructor COMMAND test_constructor)

add_executable(test_torchConversion test_torchConversion.cpp ../test_main.cpp)
target_link_libraries(test_torchConversion NSL Catch2::Catch2)
add_test(NAME test_torchConversion COMMAND test_torchConversion)

add_executable(test_randomAccess test_randomAccess.cpp ../test_main.cpp)
target_link_libraries(test_randomAccess NSL Catch2::Catch2)
add_test(NAME test_randomAccess COMMAND test_randomAccess)

add_executable(test_operatorAddition test_operatorAddition.cpp ../test_main.cpp)
target_link_libraries(test_operatorAddition NSL Catch2::Catch2)
add_test(NAME test_operatorAddition COMMAND test_operatorAddition)

add_executable(test_operatorMultiplication test_operatorMultiplication.cpp ../test_main.cpp)
target_link_libraries(test_operatorMultiplication NSL Catch2::Catch2)
add_test(NAME test_operatorMultiplication COMMAND test_operatorMultiplication)

add_executable(test_operatorCompare test_operatorCompare.cpp ../test_main.cpp)
target_link_libraries(test_operatorCompare NSL Catch2::Catch2)
add_test(NAME test_operatorCompare COMMAND test_operatorCompare)

add_executable(test_operatorAssignment test_operatorAssignment.cpp ../test_main.cpp)
target_link_libraries(test_operatorAssignment NSL Catch2::Catch2)
add_test(NAME test_operatorAssignment COMMAND test_operatorAssignment)

add_executable(test_Tensor_conj test_Tensor_conj.cpp ../test_main.cpp)
target_link_libraries(test_Tensor_conj NSL Catch2::Catch2)
add_test(NAME test_Tensor_conj COMMAND test_Tensor_conj)

add_executable(test_Tensor_mat_exp test_Tensor_mat_exp.cpp ../test_main.cpp)
target_link_libraries(test_Tensor_mat_exp NSL Catch2::Catch2)
add_test(NAME test_Tensor_mat_exp COMMAND test_Tensor_mat_exp)
