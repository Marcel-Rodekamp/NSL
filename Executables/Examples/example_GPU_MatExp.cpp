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

int main(){
    // Set a matrix size
    const std::size_t N = 1000;

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

    // perform matrix exponential and measure its execution time
    auto start_gpu = std::chrono::system_clock::now();
    T_gpu.mat_exp();
    //dev.synchronize();
    auto end_gpu = std::chrono::system_clock::now();

    // =========================================================================
    // We do the same now with a CPU to see the relative execution time:

    // define an NxN Matrix of type double on the CPU
    NSL::Tensor<double> T_cpu(N,N);
    // You can also use
    // NSL::Tensor<double> T_cpu(NSL::DEVICE::CPU(),N,N);
    // to define the same tensor on the CPU
    // fill the tensor with pseudo-random numbers
    T_cpu.rand();

    // perform matrix exponential and measure its execution time
    auto start_cpu = std::chrono::system_clock::now();
    T_cpu.mat_exp();
    auto end_cpu = std::chrono::system_clock::now();
    // =========================================================================

    // Report execution time and relative speed-up
    std::cout << "GPU Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu-start_gpu).count() << "ms."
              << std::endl;
    std::cout << "CPU Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu-start_cpu).count() << "ms."
              << std::endl;
    std::cout << "Speed-up: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu-start_cpu).count()/std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu-start_gpu).count() << "."
              << std::endl;

}
