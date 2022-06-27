#include "NSL.hpp"

int main(){
    
    // This creates a Tensor which memory is located on the GPU
    NSL::Tensor<double> A(NSL::GPU(),2,2);

    std::cout << "Random Fill GPU Tensor" << std::endl;

    // All operations to this Tensor are run on GPU
    A.rand();

    // A cout copies the data to the CPU and writes it to the terminal
    std::cout << A << std::endl;

    std::cout << "Elementwise exp on GPU" << std::endl;

    // some GPU operation, CPU and GPU are now out of sync
    A.exp();

    // Synchronize the data with CPU
    // This copies the data to the CPU and creates a view Acpu
    // A is still a gpu tensor but both now contain the same data
    auto Acpu = A.to(NSL::CPU());

    std::cout << A << std::endl;
    std::cout << Acpu << std::endl;


    std::cout << "Elementwise cos on CPU" << std::endl;

    // operate on the cpu the data is out of sync again
    Acpu.cos();

    std::cout << A << std::endl;
    std::cout << Acpu << std::endl;

    std::cout << "Synchronice CPU -> GPU" << std::endl;

    // Copy the data back to GPU explicitly
    A = Acpu.to(NSL::GPU());
    
    std::cout << A << std::endl;
    std::cout << Acpu << std::endl;


}
