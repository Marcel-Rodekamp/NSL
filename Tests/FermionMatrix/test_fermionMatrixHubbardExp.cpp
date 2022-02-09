#include "complex.hpp"
#include "../test.hpp"
#include "Tensor/tensor.hpp"
#include "Lattice/Implementations/ring.hpp"
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"


using size_type = int64_t;
template<typename T>
void test_fermionMatrixHubbardExp_M(const size_type size0, const size_type size1) {

    //NSL::TimeTensor<T> phi(const size_t & size0, const SizeType &... sizes);
    //NSL::TimeTensor<T> psi(const size_t & size0, const SizeType &... sizes);

    //hardcoding the calculation done in the method M of fermionMatrixHubbardExp class
    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1);
    NSL::TimeTensor<NSL::complex<T>> psi(size0, size1);     
    psi(0,0) = 1.;
    psi(0,1) = 1.;

    NSL::Lattice::Ring<T> r(size1);
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi);
    NSL::complex<T> I ={0,1};

    // apply kronecker delta
    NSL::TimeTensor<NSL::complex<T>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::TimeTensor<NSL::complex<T>> out =  (NSL::LinAlg::mat_vec(
       
        r.exp_hopping_matrix(/*delta=(beta/Nt) */0.1),
        ((phi*I).exp() * psiShift).transpose()
    ));

    // anti-periodic boundary condition
    out.transpose();
    out.slice(0,0,1)*=-1;
    NSL::TimeTensor<NSL::complex<T>> result = psi - out;

    //TEST  
    REQUIRE(result.real().dim() == M.M(psi).real().dim());
    REQUIRE(result.imag().dim() == M.M(psi).imag().dim());

    for (int i=0; i<size0; i++) {
        for (int j=0; j<size1; j++) {
             REQUIRE(result(i,j)==M.M(psi)(i,j)); 

    }}  
}

template<typename T>
void test_fermionMatrixHubbardExp_Mdagger(const size_type size0, const size_type size1) {

    //NSL::TimeTensor<T> phi(const size_t & size0, const SizeType &... sizes);
    //NSL::TimeTensor<T> psi(const size_t & size0, const SizeType &... sizes);

    //hardcoding the calculation done in the method Mdagger of fermionMatrixHubbardExp class
    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1);
    NSL::TimeTensor<NSL::complex<T>> psi(size0, size1);     
    psi(0,0) = 1.;
    psi(0,1) = 1.;

    NSL::Lattice::Ring<T> r(size1);
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi);
    NSL::complex<T> min_I ={0,-1};

    // apply kronecker delta
    NSL::TimeTensor<NSL::complex<T>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::TimeTensor<NSL::complex<T>> out =  (NSL::LinAlg::mat_vec(
        ((phi*min_I).exp()),
        r.exp_hopping_matrix(/*delta=(beta/Nt) */0.1)
        
    )) * psiShift;

    // anti-periodic boundary condition

    out.slice(0,0,1)*=-1;
    NSL::TimeTensor<NSL::complex<T>> result = psi - out;

    //TEST  
    REQUIRE(result.real().dim() == M.Mdagger(psi).real().dim());
    REQUIRE(result.imag().dim() == M.Mdagger(psi).imag().dim());

    for (int i=0; i<size0; i++) {
        for (int j=0; j<size1; j++) {
             REQUIRE(result(i,j)==M.Mdagger(psi)(i,j)); 

    }}  
}


template<typename T>
void test_fermionMatrixHubbardExp_MMdagger(const size_type size0, const size_type size1) {

    //NSL::TimeTensor<T> phi(const size_t & size0, const SizeType &... sizes);
    //NSL::TimeTensor<T> psi(const size_t & size0, const SizeType &... sizes);

    //hardcoding the calculation done in the method MMdagger of fermionMatrixHubbardExp class
    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1);
    NSL::TimeTensor<NSL::complex<T>> psi(size0, size1);     
    psi(0,0) = 1.;
    psi(0,1) = 1.;

    NSL::Lattice::Ring<T> r(size1);
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi);
    

    // apply kronecker delta
    NSL::TimeTensor<NSL::complex<T>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::TimeTensor<NSL::complex<T>> out =  M.M(psi) + M.Mdagger(psi)+ NSL::LinAlg::mat_vec((r.exp_hopping_matrix(0.1))*
        (r.exp_hopping_matrix(0.1)), NSL::LinAlg::mat_transpose(psi)).
    transpose();


    NSL::TimeTensor<NSL::complex<T>> result = out - psi;

    //TEST  
    REQUIRE(result.real().dim() == M.MMdagger(psi).real().dim());
    REQUIRE(result.imag().dim() == M.MMdagger(psi).imag().dim());

    for (int i=0; i<size0; i++) {
        for (int j=0; j<size1; j++) {
             REQUIRE(result(i,j)==M.MMdagger(psi)(i,j)); 

    }}  
}

template<typename T>
void test_fermionMatrixHubbardExp_MdaggerM(const size_type size0, const size_type size1) {

    //NSL::TimeTensor<T> phi(const size_t & size0, const SizeType &... sizes);
    //NSL::TimeTensor<T> psi(const size_t & size0, const SizeType &... sizes);

    //hardcoding the calculation done in the method MdaggerM of fermionMatrixHubbardExp class
    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1);
    NSL::TimeTensor<NSL::complex<T>> psi(size0, size1);     
    psi(0,0) = 1.;
    psi(0,1) = 1.;

    NSL::Lattice::Ring<T> r(size1);
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi);
    NSL::complex<T> min_I ={0,-1}, I = {0,1};
    

    // apply kronecker delta
    NSL::TimeTensor<NSL::complex<T>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::TimeTensor<NSL::complex<T>> out =  M.M(psi) + M.Mdagger(psi)+ ((((phi*min_I).exp()).transpose())  *
        NSL::LinAlg::mat_vec((r.exp_hopping_matrix(0.1))*
        (r.exp_hopping_matrix(0.1)), NSL::LinAlg::mat_transpose((phi*I).exp()*psi))).
    transpose();



    NSL::TimeTensor<NSL::complex<T>> result = out - psi;

    //TEST  
    REQUIRE(result.real().dim() == M.MdaggerM(psi).real().dim());
    REQUIRE(result.imag().dim() == M.MdaggerM(psi).imag().dim());

    for (int i=0; i<size0; i++) {
        for (int j=0; j<size1; j++) {
             REQUIRE(result(i,j)==M.MdaggerM(psi)(i,j)); 

    }}  
}

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
    REQUIRE(res1 <= limit);
    REQUIRE(res2 <= limit);


    
}
//Test cases
TEST_CASE( "fermionMatrixHubbardExp: M", "[fermionMatrixHubbardExp, M]" ) {

    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    test_fermionMatrixHubbardExp_M<float>(size_0, size_1);
    test_fermionMatrixHubbardExp_M<double>(size_0, size_1);

}

TEST_CASE( "fermionMatrixHubbardExp: Mdagger", "[fermionMatrixHubbardExp, Mdagger]" ) {

    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    test_fermionMatrixHubbardExp_Mdagger<float>(size_0, size_1);
    test_fermionMatrixHubbardExp_Mdagger<double>(size_0, size_1);


}

TEST_CASE( "fermionMatrixHubbardExp: MMdagger", "[fermionMatrixHubbardExp, MMdagger]" ) {

    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    test_fermionMatrixHubbardExp_MMdagger<double>(size_0, size_1);
    test_fermionMatrixHubbardExp_MMdagger<float>(size_0, size_1);

}

TEST_CASE( "fermionMatrixHubbardExp: MdaggerM", "[fermionMatrixHubbardExp, MdaggerM]" ) {

    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    test_fermionMatrixHubbardExp_MdaggerM<double>(size_0, size_1);
    test_fermionMatrixHubbardExp_MdaggerM<float>(size_0, size_1);

}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM", "[fermionMatrixHubbardExp, logDetM]" ) {
    
    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM<TestType>(size_0, size_1);
    
    

}