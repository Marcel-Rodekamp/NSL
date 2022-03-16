#include <iostream>
#include "Tensor/tensor2.hpp"

template<typename Type, typename SizeType, NSL::isType<SizeType> ... SizeTypes >
void construct_tensors(const SizeType & size0, const SizeTypes & ... sizes){
    std::array<SizeType,sizeof...(sizes)> sizes_array{sizes...};

    // default constructor
    // constructs single element tensor with value 0
    NSL::Tensor<Type> default_constructed;

    // D-dimensional constructor
    // Constructs a vector with size0 elements
    NSL::Tensor<Type> vector(size0);

    // D-dimensional constructor
    // Constructs a vector with size0 x sizes[0] elements
    NSL::Tensor<Type> matrix(size0,sizes_array[0]);

    // D-dimensional constructor
    // Constructs a vector with size0 x sizes[0] elements
    NSL::Tensor<Type> tensorD(size0,sizes...);

    std::cout << tensorD << std::endl;
}

int main(int argc, char ** argv){
    const std::size_t N0 = 10;
    const std::size_t N1 = 10;
    const std::size_t N2 = 10;
    construct_tensors<float>(N0,N1,N2);

}