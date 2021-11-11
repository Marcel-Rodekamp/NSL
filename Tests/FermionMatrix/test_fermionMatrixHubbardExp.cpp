#include <iostream>

#include "../test.hpp"
#include <typeinfo>
#include "complex.hpp"
#include "Tensor/tensor.hpp"
#include "Lattice/Implementations/ring.hpp"
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"

void test_fermionMatrixHubbardExp() {

    NSL::complex<double> iota = {0,1};
    NSL::TimeTensor<NSL::complex<double>> phi(16,2);
    NSL::TimeTensor<NSL::complex<double>> psi(16,2);
    psi(0,0) = 1.;
    psi(0,1) = 1.;
    NSL::Lattice::Ring<double> r(2);
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi);


//printing out various values for testing
    std::cout<<phi*iota<<std::endl;
    std::cout<<NSL::LinAlg::shift(psi,1)<<std::endl;
    std::cout<<phi.exp()<<std::endl;
    std::cout<<(phi.exp())*(NSL::LinAlg::shift(psi,1))<<std::endl;
    std::cout<<psi<<std::endl;
    std::cout<<r.hopping_matrix(0.1).real()<<std::endl;
    std::cout<<r.exp_hopping_matrix(0.1).real()<<std::endl;
    std::cout<<r.exp_hopping_matrix(0.1).imag()<<std::endl;
    std::cout << M.M(psi).real() << std::endl;
    std::cout << M.M(psi).imag() << std::endl;

    std::cout<<NSL::LinAlg::mat_vec(r.exp_hopping_matrix(0.1),((phi.exp())*(NSL::LinAlg::shift(psi,1))).transpose())<<std::endl;


//trying with arbtrary a and b (phi and psi)
NSL::TimeTensor<NSL::complex<double>> a(4,4);

NSL::TimeTensor<NSL::complex<double>> b(4,4);

for (int i=0; i<4; i++)
{ for (int j=0; j<4; j++)
{a(i,j)=i+j;
b(i,j)=j-i;
}}

//printing out various values for testing
std::cout<<a.exp()<<std::endl;
std::cout<<b<<std::endl;
std::cout<<NSL::LinAlg::shift(b,1)<<std::endl;
std::cout<<a.exp() * b<<std::endl; 


std::cout<<NSL::LinAlg::shift(b,-1)<<std::endl;
std::cout<<(a.exp())*(NSL::LinAlg::shift(b,-1))<<std::endl;
std::cout<<((a.exp())*(NSL::LinAlg::shift(b,-1))).imag()<<std::endl;


NSL::TimeTensor<NSL::complex<double>> c(1,4);

NSL::TimeTensor<NSL::complex<double>> d(1,4);
for (int j=0; j<4; j++){
c(0,j)=j+5;
d(0,j)=j;}

std::cout<<c.exp()<<std::endl;
std::cout<<d<<std::endl;
std::cout<<NSL::LinAlg::shift(d,1)<<std::endl;
std::cout<<c.exp() * d<<std::endl; 


}

//Not exactly a test case, just printing out values
REAL_NSL_TEST_CASE( "FermionMatrix: fermionMatrixHubbardExp", "[FermionMatrix, fermionMatrixHubbardExp]"){
test_fermionMatrixHubbardExp();}


//