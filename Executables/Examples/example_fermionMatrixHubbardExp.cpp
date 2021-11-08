#include <iostream>
#include "complex.hpp"
#include "Tensor/tensor.hpp"
#include "Lattice/Implementations/ring.hpp"
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"

int main(){

//    torch::Tensor m = torch::zeros({2,2}, torch::TensorOptions().dtype<double>());
//    torch::Tensor v = torch::zeros({2}, torch::TensorOptions().dtype<NSL::complex<double>>());
//
//    std::cout<< torch::matmul(m,v) << std::endl;

    NSL::TimeTensor<NSL::complex<double>> phi(16,2);
    NSL::TimeTensor<NSL::complex<double>> psi(16,2);
    psi(0,0) = 1.;
    psi(0,1) = 1.;
    NSL::Lattice::Ring<double> r(2);
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi);

    std::cout << M.M(psi).real() << std::endl;
    std::cout << M.M(psi).imag() << std::endl;

    return EXIT_SUCCESS;
}
