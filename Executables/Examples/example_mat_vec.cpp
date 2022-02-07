#include <iostream>
#include "complex.hpp"
#include "Tensor/tensor.hpp"
#include "Lattice/Implementations/ring.hpp"
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"
#include "LinAlg/mat_vec.hpp"



int main(){


    NSL::TimeTensor<NSL::complex<double>> phi(10,2), ex(2,2);

    ex.rand();
    phi.rand();
    
    //mat_vec
    std::cout<<NSL::LinAlg::mat_vec(ex, phi.transpose())<<std::endl;
    //mat_mul
    std::cout<<ex.mat_mul(phi)<<std::endl;

    return EXIT_SUCCESS;
}
