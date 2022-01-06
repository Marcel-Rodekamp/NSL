#include <iostream>
#include <chrono>
#include "Tensor/tensor.hpp"

int main(){
    // Set a matrix size
    const std::size_t N = 5000;

    // check that a GPU is available
    // this is a runtime check
    if(! NSL::DEVICE::GPU::is_available()){
        std::cerr << "No GPU available" << std::endl;
        return EXIT_FAILURE;
    }

    // Define a device object
    NSL::DEVICE::GPU dev;

    // define an NxN Matrix of type double on the GPU
    NSL::Tensor<double> T_gpu(dev,N,N);
    // fill the tensor with pseudo-random numbers
    T_gpu.rand();
}
