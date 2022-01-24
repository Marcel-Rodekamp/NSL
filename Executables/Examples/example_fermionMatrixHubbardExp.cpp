#include <iostream>
#include "complex.hpp"
#include "Tensor/tensor.hpp"
#include "Lattice/Implementations/ring.hpp"
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"
#include "LinAlg/mat_conj.hpp"
#include "LinAlg/mat_exp.hpp"

int main(){


    NSL::TimeTensor<NSL::complex<double>> phi(16,2);
    NSL::TimeTensor<NSL::complex<double>> psi(16,2);
    NSL::TimeTensor<double> r_tens(16,2);
    //psi(0,0) = {1,1};
    //psi(0,1) = {1,1};

    //phi(0,0) = {1,1};
    //phi(0,1) = {1,1};
    phi.rand();
    psi.rand();
    r_tens.rand();
    NSL::complex<double> iota = {0,1};
    double min_one =-1;

    NSL::TimeTensor<NSL::complex<double>> out = NSL::LinAlg::mat_vec(psi.transpose(),r_tens);
    std::cout<<out<<std::endl;

    NSL::Lattice::Ring<double> r(2); 
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<double>> M(&r,psi);


    //std::cout<<(phi*iota).exp().imag()<<std::endl;
    //std::cout<<((NSL::LinAlg::adjoint(phi)*(-iota)).exp()).real()<<std::endl;
    std::cout<<M.M(psi).real()<<std::endl;
    std::cout<<M.M(psi).imag()<<std::endl;


    std::cout<<M.MdaggerM(psi).real()<<std::endl;
    std::cout<<M.MdaggerM(psi).imag()<<std::endl; 


    std::cout<<M.MMdagger(psi).real()<<std::endl;
    std::cout<<M.MMdagger(psi).imag()<<std::endl;

    std::cout<<M.Mdagger(psi).real()<<std::endl;
    std::cout<<M.Mdagger(psi).imag()<<std::endl;

    return EXIT_SUCCESS;
}
