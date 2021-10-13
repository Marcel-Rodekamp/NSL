#include <iostream>
#include "Tensor/tensor.hpp"

int main(int argc, char ** argv){
    NSL::Tensor<NSL::complex<double>> Tc(2);
    NSL::Tensor<double> Tr(2);

    for(int i = 0; i < 2; ++i){
        Tc(i) = NSL::complex<double>(i+1,i+1);
        Tr(i) = i+1;
    }

    Tc.conj();
    Tr.conj();

    std::cout << Tc << std::endl;
    std::cout << Tr << std::endl;

}