#include "complex.hpp"
#include "../test.hpp"
#include "Tensor/tensor.hpp"
#include "Lattice/Implementations/ring.hpp"
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"


using size_type = int64_t;

void test_fermionMatrixHubbardExp(const size_type size0, const size_type size1) {

    //NSL::TimeTensor<T> phi(const size_t & size0, const SizeType &... sizes);
    //NSL::TimeTensor<T> psi(const size_t & size0, const SizeType &... sizes);
    NSL::TimeTensor<NSL::complex<double>> phi(size0, size1);
    NSL::TimeTensor<NSL::complex<double>> psi(size0, size1);     
    psi(0,0) = 1.;
    psi(0,1) = 1.;

    NSL::Lattice::Ring<double> r(size1);
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi);
    NSL::complex<double> I ={0,1};

    // apply kronecker delta
    NSL::TimeTensor<NSL::complex<double>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::TimeTensor<NSL::complex<double>> out =  (NSL::LinAlg::mat_vec(
       
        r.exp_hopping_matrix(/*delta=(beta/Nt) */0.1),
        ((phi*I).exp() * psiShift).transpose()
    ));

    // anti-periodic boundary condition

    out.slice(0,0,1)*=-1;
    NSL::TimeTensor<NSL::complex<double>> result = psi - (out).transpose();

    //TEST  
    REQUIRE(result.real().dim() == M.M(psi).real().dim());
    REQUIRE(result.imag().dim() == M.M(psi).imag().dim());

    for (int i=0; i<size0; i++) {
        for (int j=0; j<size1; j++) {
             REQUIRE(result(i,j)==M.M(psi)(i,j)); 


    }}
  
}

//Test cases

NSL_TEST_CASE( "fermionMatrixHubbardExp: M", "[fermionMatrixHubbardExp, M]" ) {

    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    test_fermionMatrixHubbardExp(size_0, size_1);
}