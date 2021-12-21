#include <iostream>
#include <chrono>
#include "Tensor/tensor.hpp"

int main(){
    const std::size_t N = 5000;

    NSL::Tensor<double> T_gpu(NSL::DEVICE::GPU(),N,N);
//    NSL::Tensor<double> T_cpu(NSL::DEVICE::CPU(),N,N);
    NSL::Tensor<double> T_cpu(N,N);

    std::cout << "GPU...";
    auto start_gpu = std::chrono::system_clock::now();
    T_gpu.mat_exp();
    auto end_gpu = std::chrono::system_clock::now();
    std::cout << " done." << std::endl;
    T_gpu.to(NSL::DEVICE::CPU());
    std::cout << T_gpu(0,0) << std::endl;
    std::cout << "CPU...";
    auto start_cpu = std::chrono::system_clock::now();
    T_cpu.mat_exp();
    auto end_cpu = std::chrono::system_clock::now();
    std::cout << " done." << std::endl;
    std::cout << T_cpu(0,0) << std::endl;

    std::cout << "GPU Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu-start_gpu).count() << "ms." << std::endl;
    std::cout << "CPU Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu-start_cpu).count() << "ms." << std::endl;

}
