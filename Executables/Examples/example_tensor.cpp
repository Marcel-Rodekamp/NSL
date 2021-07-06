#include <iostream>
#include "Tensor/tensor.hpp"

#ifdef __CUDACC__

#include "Tensor/tensor.cuh"

#endif // __CUDACC__

template<typename Type>
__global__ void print_tensor(Type * data, std::size_t N){
    auto i = threadIdx.x;

    if (i != N-1 && i < N){
        printf("%f, ",data[i]);
    } else {
        printf("%f",data[i]);
    }
}

int main(){
    std::cout << "================================================================================" << std::endl
              << "Showing Tensor Interface" << std::endl
              << "================================================================================" << std::endl;

    const std::size_t N = 5;

    // cpu tensor
    NSL::Tensor<double, false> tensor_cpu(N);
#ifdef __CUDACC__
    // gpu tensor
    NSL::Tensor<double, true> tensor_gpu(N);
#endif // __CUDACC__

    std::cout << "CPU Tensor after initialization:\n T = [";
    for(std::size_t i = 0; i < N; ++i){
        if (i != N-1) {
            std::cout << tensor_cpu[i] << ", ";
        } else {
            std::cout << tensor_cpu[i];
        }
    }
    std::cout << "]" << std::endl;

#ifdef __CUDACC__

    std::cout << "GPU Tensor after initialization print from CPU:\n T = [";
    for(std::size_t i = 0; i < N; ++i){
        if (i != N-1) {
            std::cout << tensor_cpu[i] << ", ";
        } else {
            std::cout << tensor_cpu[i];
        }
    }
    std::cout << "]" << std::endl;


    std::cout << "GPU Tensor after initialization print from GPU:\n T = [";
    print_tensor<<<1,N>>>(tensor_gpu.data(),N);
    cudaDeviceSynchronize();
    std::cout << "]" << std::endl;


#endif // __CUDACC__

}