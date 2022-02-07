#include "complex.hpp"
#include "../test.hpp"
#include "Tensor/tensor.hpp"
#include "Lattice/Implementations/ring.hpp"
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"
#include <iostream>
#include<limits>
#include<math.h>
using size_type = int64_t;
template<typename T>
void test_logDetM(const size_type size0, const size_type size1) {

    auto limit = std::pow(10, 2-std::numeric_limits<T>::digits10);
    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1), phiShift(size0, size1);
    phi.rand();
    NSL::Lattice::Ring<T> ring(size1);
    
    //FermionMatrixHubbardExp Object M for phi and ring lattice
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<T>> M(&ring,phi);
    
    //FermionMatrixHubbardExp Object Mshift for phiShift and ring lattice
    phiShift=NSL::LinAlg::shift(phi,4);
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<T>> Mshift(&ring,phiShift);
    
    //TEST  
    auto res1 = fabs(M.logDetM().real() - Mshift.logDetM().real());
    auto res2 = fabs(M.logDetM().imag() - Mshift.logDetM().imag());
    std::cout<<"Limit: "<<limit<<std::endl;
    std::cout<<"Res: "<<res1<<std::endl;
    REQUIRE(res1 <= limit);
    REQUIRE(res2 <= limit);


    
}

//Test cases
REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM", "[fermionMatrixHubbardExp, logDetM]" ) {
    
    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM<TestType>(size_0, size_1);
    
    

}