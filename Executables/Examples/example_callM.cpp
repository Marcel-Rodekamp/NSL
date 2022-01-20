#include <iostream>
#include "complex.hpp"
#include "Tensor/tensor.hpp"
#include "Lattice/Implementations/ring.hpp"
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"
#include "LinAlg/mat_conj.hpp"
#include "LinAlg/mat_exp.hpp"

int main(){

    //Initializing
    NSL::TimeTensor<NSL::complex<double>> phi(2,2);
    NSL::TimeTensor<NSL::complex<double>> psi(2,2);
    phi.rand();
    psi.rand();
    NSL::complex<double> iota = {0,1};
    double min_one =-1;    
    NSL::Lattice::Ring<double> r(2); 

    //calling the function M
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<double>> M(&r,psi);


    return EXIT_SUCCESS;
}
