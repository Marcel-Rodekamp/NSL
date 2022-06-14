add_executable(test_fermionMatrixHubbardExp test_fermionMatrixHubbardExp.cpp ../test_main.cpp)
target_link_libraries(test_fermionMatrixHubbardExp NSL Catch2::Catch2)
add_test(NAME test_fermionMatrixHubbardExp COMMAND test_fermionMatrixHubbardExp)

add_executable(test_FermionMatrix_hermitian test_FermionMatrix_hermitian.cpp ../test_main.cpp)
target_link_libraries(test_FermionMatrix_hermitian NSL Catch2::Catch2)
add_test(NAME test_FermionMatrix_hermitian COMMAND test_FermionMatrix_hermitian)