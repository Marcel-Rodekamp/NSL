#include <iostream>
#include "complex.hpp"
#include "Tensor/tensor.hpp"
#include "Lattice/Implementations/ring.hpp"
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"
#include "Linalg/mat_conj.hpp"
#include "Linalg/mat_exp.hpp"

int main(){


    NSL::TimeTensor<NSL::complex<double>> phi(16,2);
    NSL::TimeTensor<NSL::complex<double>> psi(16,2);
    //psi(0,0) = {1,1};
    //psi(0,1) = {1,1};

    //phi(0,0) = {1,1};
    //phi(0,1) = {1,1};
    phi.rand();
    psi.rand();
    NSL::complex<double> iota = {0,1};
    double min_one =-1;
    
    NSL::Lattice::Ring<double> r(2); 
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<double>> M(&r,psi);

    std::cout<<NSL::LinAlg::exp(psi)<<std::endl;
    std::cout<<psi.exp()<<std::endl;
    //std::cout<<(phi*iota).exp().imag()<<std::endl;
    //std::cout<<((NSL::LinAlg::adjoint(phi)*(-iota)).exp()).real()<<std::endl;
    std::cout<<M.M(psi).real()<<std::endl;
    std::cout<<M.M(psi).imag()<<std::endl;

    std::cout<<M.Mdagger(psi).real()<<std::endl;
    std::cout<<M.Mdagger(psi).imag()<<std::endl;

    std::cout<<M.MMdagger(psi).real()<<std::endl;
    std::cout<<M.MMdagger(psi).imag()<<std::endl;

    std::cout<<M.MdaggerM(psi).real()<<std::endl;
    std::cout<<M.MdaggerM(psi).imag()<<std::endl; 
    return EXIT_SUCCESS;
}