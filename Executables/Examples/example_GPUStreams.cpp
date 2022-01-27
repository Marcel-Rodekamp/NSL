//! Example GPU Matrix Exponential
/*!
 * This file contains a short example on the usage of GPU tensors.
 * 1. Construction of Tensors on the GPU
 * 2. Execute a kernel on the GPU
 * */

// C++ STL IO
// https://en.cppreference.com/w/cpp/header/iostream
#include <iostream>

// C++ STL Timing
// https://en.cppreference.com/w/cpp/header/chrono
#include <chrono>

// Include NSL
#include "NSL.hpp"
// alternatively you could include
// #include "Tensor/tensor.hpp"
// #include "gpu.hpp"

void CUDA_Stream_GetSet(const std::size_t N){
     // Define a device object
    NSL::DEVICE::GPU dev;

    // define a CPU object
    NSL::DEVICE::CPU cpu;

    // define an NxN Matrix of type double on the GPU
    NSL::Tensor<double> T_gpu(dev,N,N);
    NSL::Tensor<double> T_cpu(dev,N,N);
    // fill the tensor with pseudo-random numbers
    T_gpu.rand();

    // get a stream from the pool for the device dev
    NSL::DEVICE::Stream copyStream = dev.Stream();

    auto start = std::chrono::system_clock::now();

    // set the stream for copying
    dev.CurrentStream(copyStream);

    // perform some operation using the current stream
    auto start_copy = std::chrono::system_clock::now();
    //T_cpu = NSL::to(T_gpu, cpu);
    T_gpu.to(cpu, true);
    auto end_copy = std::chrono::system_clock::now();

    // synchronize the streams
    dev.synchronize();

    // reset to default stream
    dev.CurrentStream( dev.DefaultStream() );
    
    auto end = std::chrono::system_clock::now();

    //std::cout << "T_cpu = " << std::endl
    //          << T_cpu.slice(0,0,2) << std::endl;

    std::cout << "Copy Call Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_copy-start_copy).count() << "ms."
              << std::endl;
    std::cout << "Copy Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms."
              << std::endl;

}


int main(){
    // Set a matrix size
    const std::size_t N = 5000;

    // check that a GPU is available
    // this is a runtime check
    if(! NSL::DEVICE::GPU::is_available()){
        std::cerr << "No GPU available" << std::endl;
        return EXIT_FAILURE;
    }
    
    CUDA_Stream_GetSet(N);
}
