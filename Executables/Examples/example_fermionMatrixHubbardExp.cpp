#include <iostream>
#include "complex.hpp"
#include "Tensor/tensor.hpp"
#include "Lattice/Implementations/ring.hpp"
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"
#include "LinAlg/mat_conj.hpp"
#include "LinAlg/mat_exp.hpp"
#include "Tensor/Matrices/matricesBase.hpp"

#include "LinAlg/mat_vec.hpp"
#include "LinAlg/mat_mul.hpp"
#include "LinAlg/mat_trans.hpp"
#include "LinAlg/det.hpp"
#include "LinAlg/mat_inv.hpp"
#include "LinAlg/matrix.hpp"
#include "math.h"



int main(){
    NSL::TimeTensor<NSL::complex<double>> phi(16,2);
    NSL::TimeTensor<NSL::complex<double>> psi(16,2);

    phi.rand();
    psi.rand();
    

    NSL::Lattice::Ring<double> r(2); 
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<double>> M(&r,psi);

    std::cout<<M.M(psi).real()<<std::endl;
    std::cout<<M.M(psi).imag()<<std::endl; 

    std::cout<<M.MdaggerM(psi).real()<<std::endl;
    std::cout<<M.MdaggerM(psi).imag()<<std::endl; 


    std::cout<<M.MMdagger(psi).real()<<std::endl;
    std::cout<<M.MMdagger(psi).imag()<<std::endl;

    std::cout<<M.Mdagger(psi).real()<<std::endl;
    std::cout<<M.Mdagger(psi).imag()<<std::endl;

    std::cout<<M.logDetM()<<std::endl;

    std::cout<<M.logDetMdagger()<<std::endl;


   


    return EXIT_SUCCESS;
}
