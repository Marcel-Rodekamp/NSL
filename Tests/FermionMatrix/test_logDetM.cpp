#include "complex.hpp"
#include "../test.hpp"
#include "Tensor/tensor.hpp"
#include "Lattice/Implementations/ring.hpp"
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"
#include <iostream>


using size_type = int64_t;
template<typename T>
void test_logDetM(const size_type size0, const size_type size1) {

    
    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1), phiShift(size0, size1);
 
    //phi constant in time
    for(int i=0;i<10;i++){
        for(int j=0;j<2;j++){
            phi(i,j) += NSL::complex<T>(i+1,i+2);

        }
    }

    NSL::Lattice::Ring<T> r(size1);
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<T>> M(&r,phi);
    
    phiShift=NSL::LinAlg::shift(phi,4);
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<T>> Mshift(&r,phiShift);

    //TEST  
    REQUIRE(M.logDetM().real() == Mshift.logDetM().real());
    REQUIRE(M.logDetM().imag() == Mshift.logDetM().imag());

    
}

//Test cases
TEST_CASE( "fermionMatrixHubbardExp: logDetM", "[fermionMatrixHubbardExp, logDetM]" ) {

    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    test_logDetM<float>(size_0, size_1);
    test_logDetM<double>(size_0, size_1);

}