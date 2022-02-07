add_executable(example_tensor example_tensor.cpp)
target_link_libraries(example_tensor NSL)

add_executable(example_fermionMatrixHubbardExp example_fermionMatrixHubbardExp.cpp)
target_link_libraries(example_fermionMatrixHubbardExp NSL)

add_executable(example_callM example_callM.cpp)
target_link_libraries(example_callM NSL)

add_executable(example_mat_vec example_mat_vec.cpp)
target_link_libraries(example_mat_vec NSL)
