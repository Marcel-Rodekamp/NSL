#include "complex.hpp"
#include "../test.hpp"
#include "Tensor/tensor.hpp"
#include "Lattice/Implementations/ring.hpp"
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"
#include "math.h"
#include <time.h> 


using size_type = int64_t;

//Test for the function M(psi)
template<typename T>
void test_fermionMatrixHubbardExp_M(const size_type size0, const size_type size1) {

    //NSL::TimeTensor<T> phi(const size_t & size0, const SizeType &... sizes);
    //NSL::TimeTensor<T> psi(const size_t & size0, const SizeType &... sizes);

    //setting precision
    auto limit =  10*std::numeric_limits<T>::epsilon();

    //hardcoding the calculation done in the method M of fermionMatrixHubbardExp class
    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1);
    NSL::TimeTensor<NSL::complex<T>> psi(size0, size1);     
    psi.rand();

    NSL::Lattice::Ring<T> r(size1);
    //delta= beta/Nt
    double delta = 2.0/size0;
    //FermionMatrixHubbardExp Object M
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi,2.0);
    NSL::complex<T> I ={0,1};

    // apply kronecker delta
    NSL::TimeTensor<NSL::complex<T>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::TimeTensor<NSL::complex<T>> out =  (NSL::LinAlg::mat_vec(       
        r.exp_hopping_matrix(/*delta=(beta/Nt) */delta),
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
            auto res1 = fabs(result(i,j).real() - M.M(psi)(i,j).real());
            auto res2 = fabs(result(i,j).imag() - M.M(psi)(i,j).imag());
            REQUIRE(res1 <= limit);
            REQUIRE(res2 <= limit);
            

    }}  

}

//Test for the function Mdagger(psi)
template<typename T>
void test_fermionMatrixHubbardExp_Mdagger(const size_type size0, const size_type size1) {

    //setting precision
    auto limit =  10*std::numeric_limits<T>::epsilon();

    //hardcoding the calculation done in the method Mdagger of fermionMatrixHubbardExp class
    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1);
    NSL::TimeTensor<NSL::complex<T>> psi(size0, size1);     
    psi.rand();

    NSL::Lattice::Ring<T> r(size1);
    //delta=beta/Nt
    double delta = 2.0/size0;
    //FermionMatrixHubbardExp Object M
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi,2.0);
    NSL::complex<T> min_I ={0,-1};

    // apply kronecker delta
    NSL::TimeTensor<NSL::complex<T>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::TimeTensor<NSL::complex<T>> out =  (NSL::LinAlg::mat_vec(
        ((phi*min_I).exp()),
        r.exp_hopping_matrix(/*delta=(beta/Nt) */delta)
        
    )) * psiShift;

    // anti-periodic boundary condition
    out.slice(0,0,1)*=-1;
    NSL::TimeTensor<NSL::complex<T>> result = psi - out;

    //TEST  
    REQUIRE(result.real().dim() == M.Mdagger(psi).real().dim());
    REQUIRE(result.imag().dim() == M.Mdagger(psi).imag().dim());

    for (int i=0; i<size0; i++) {
        for (int j=0; j<size1; j++) {
            auto res1 = fabs(result(i,j).real() - M.Mdagger(psi)(i,j).real());
            auto res2 = fabs(result(i,j).imag() - M.Mdagger(psi)(i,j).imag());
            REQUIRE(res1 <= limit);
            REQUIRE(res2 <= limit);
              

    }}   
}

//Test for the function MMdagger(psi)
template<typename T>
void test_fermionMatrixHubbardExp_MMdagger(const size_type size0, const size_type size1) {

    auto limit =  10*std::numeric_limits<T>::epsilon();

    //hardcoding the calculation done in the method MMdagger of fermionMatrixHubbardExp class
    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1);
    NSL::TimeTensor<NSL::complex<T>> psi(size0, size1);     
    psi.rand();

    NSL::Lattice::Ring<T> r(size1);
    //delta=beta/Nt
    double delta = 2.0/size0;
    //FermionMatrixHubbardExp Object M
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi,2.0);
    
    // apply kronecker delta
    NSL::TimeTensor<NSL::complex<T>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::TimeTensor<NSL::complex<T>> out =  M.M(psi) + M.Mdagger(psi)+ NSL::LinAlg::mat_vec((r.exp_hopping_matrix(delta))*
        (r.exp_hopping_matrix(delta)), NSL::LinAlg::mat_transpose(psi)).
    transpose();

    NSL::TimeTensor<NSL::complex<T>> result = out - psi;

    //TEST  
    REQUIRE(result.real().dim() == M.MMdagger(psi).real().dim());
    REQUIRE(result.imag().dim() == M.MMdagger(psi).imag().dim());

    for (int i=0; i<size0; i++) {
        for (int j=0; j<size1; j++) {
            auto res1 = fabs(result(i,j).real() - M.MMdagger(psi)(i,j).real());
            auto res2 = fabs(result(i,j).imag() - M.MMdagger(psi)(i,j).imag());
            REQUIRE(res1 <= limit);
            REQUIRE(res2 <= limit);
              

    }}  
}

//Test for the function MdaggerM(psi)
template<typename T>
void test_fermionMatrixHubbardExp_MdaggerM(const size_type size0, const size_type size1) {

    auto limit =  10*std::numeric_limits<T>::epsilon();

    //hardcoding the calculation done in the method MdaggerM of fermionMatrixHubbardExp class
    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1);
    NSL::TimeTensor<NSL::complex<T>> psi(size0, size1);     
    psi.rand();

    NSL::Lattice::Ring<T> r(size1);
    //delta=beta/Nt
    double delta = 2.0/size0;
    //FermionMatrixHubbardExp Object M
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi,2.0);
    NSL::complex<T> min_I ={0,-1}, I={0,1};
    
    // apply kronecker delta
    NSL::TimeTensor<NSL::complex<T>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::TimeTensor<NSL::complex<T>> out =  M.M(psi) + M.Mdagger(psi)+ ((((phi*min_I).exp()).transpose())  *
        NSL::LinAlg::mat_vec((r.exp_hopping_matrix(delta))*
        (r.exp_hopping_matrix(delta)), NSL::LinAlg::mat_transpose((phi*I).exp()*psi))).
    transpose();

    NSL::TimeTensor<NSL::complex<T>> result = out - psi;

    //TEST  
    REQUIRE(result.real().dim() == M.MdaggerM(psi).real().dim());
    REQUIRE(result.imag().dim() == M.MdaggerM(psi).imag().dim());

    for (int i=0; i<size0; i++) {
        for (int j=0; j<size1; j++) {
            auto res1 = fabs(result(i,j).real() - M.MdaggerM(psi)(i,j).real());
            auto res2 = fabs(result(i,j).imag() - M.MdaggerM(psi)(i,j).imag());
            REQUIRE(res1 <= limit);
            REQUIRE(res2 <= limit);
              

    }} 
}

//Test for the function logDetM() (shift in phi)
template<typename T>
void test_logDetM_1(const size_type size0, const size_type size1) {

    //setting precision
    //auto limit = std::pow(10, 2-std::numeric_limits<T>::digits10);
    auto limit =  10*std::numeric_limits<T>::epsilon();

    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1), phiShift(size0, size1);
    phi.rand();
    NSL::Lattice::Ring<T> ring(size1);
    //delta=beta/Nt
    double delta = 0.1/size0;
    
    //FermionMatrixHubbardExp Object M for ring lattice 
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<T>> M(&ring,phi,0.1);
    
    //FermionMatrixHubbardExp Object Mshift for the shifted phi
    phiShift=NSL::LinAlg::shift(phi,4);
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<T>> Mshift(&ring,phiShift,0.1);
    
    //TEST 
    auto res1 = fabs(M.logDetM().real() - Mshift.logDetM().real());
    auto res2 = fabs(M.logDetM().imag() - Mshift.logDetM().imag());
  
    REQUIRE(res1 <= limit);
    REQUIRE(res2 <= limit);
    
}

//Test for logDetM() (adding 2*pi in one of the time slices )
template<typename T>
void test_logDetM_2(const size_type size0, const size_type size1) {

    //setting precision
    //auto limit = std::pow(10, 2-std::numeric_limits<T>::digits10);
    auto limit =  10*std::numeric_limits<T>::epsilon();
    
    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1), phiShift(size0, size1);
    phi.rand();
    NSL::Lattice::Ring<T> ring(size1);
    T delta = 0.1/size0;
    
    //FermionMatrixHubbardExp Object M 
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<T>> M(&ring,phi,0.1);
    
    //generating random time slice to add 2*pi
    srand (time(NULL));
    int t = rand() % size0;
    //pi
    NSL::complex<T> pi ={2*std::numbers::pi,0};
    //adding 2*pi at the t_th position
    for(int i=0; i< size1; i++){
        phi(t, i)=phi(t,i) + pi;
        }
    //FermionMatrixHubbardExp Object with modified phi    
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<T>> Mshift(&ring,phi,0.1);
    
    //TEST 
    auto res1 = fabs(M.logDetM().real() - Mshift.logDetM().real());
    auto res2 = fabs(M.logDetM().imag() - Mshift.logDetM().imag());
    
    REQUIRE(res1 <= limit);
    REQUIRE(res2 <= limit);


}

//Test for logDetM() when phi=0
template<typename T>
void test_logDetM_3(const size_type size0, const size_type size1) {

    //setting precision (this test fails for higher precision)
    //auto limit = std::pow(10, 3-std::numeric_limits<T>::digits10);
    T limit =  100*std::numeric_limits<T>::epsilon();

    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1); // phiShift(size0, size1);    
    NSL::Lattice::Ring<T> ring(size1);
    //delta=beta/Nt
    T delta = 0.01/size0;
    //FermionMatrixHubbardExp Object 
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<T>> M(&ring,phi,0.01);
    
    //When phi=0, logDetM = logdet(1 + exp_hopping_matrix(beta))
    NSL::TimeTensor<T> Id(size1,size1);
    NSL::complex<T> result = NSL::LinAlg::logdet(NSL::LinAlg::Matrix::Identity(Id, size1) + ring.exp_hopping_matrix(delta*size0)); //
    
    
    //TEST 
    T res1 = fabs(M.logDetM().real() - result.real());
    T res2 = fabs(M.logDetM().imag() - result.imag());
    //double ratio = fabs(1- M.logDetM().real()/result.real());
    //double maxXY = std::fmax( std::fabs(result.real()) , std::fabs(M.logDetM().real()) ) ;
    //REQUIRE(ratio <= limit);
    REQUIRE(res1 <= limit);
    REQUIRE(res2 <= limit);
    
    
    
}

//Test for logDetM() when  all the elemets in every time slice are same
template<typename T>
void test_logDetM_4(const size_type size0, const size_type size1) {

    //setting precision
    //auto limit = std::pow(10, 2-std::numeric_limits<T>::digits10);
    T limit =  10*std::numeric_limits<T>::epsilon();

    NSL::TimeTensor<NSL::complex<T>> phi(size0, size1), phisum(1,size1), Id(size1,size1);    
    NSL::Lattice::Ring<T> ring(size1);
    //delta=beta/Nt
    T delta = 0.1/size0;
    NSL::complex<T> I ={0,1};

    //setting up phi such that all the elements in a time slice are same
    int i=0, j=0;
    NSL::complex<T> sum ={0,0};
    for(i=0; i<size0; i++){
        for(j=0; j<size1; j++){
            phi(i,j) = {(2.0*i) +0.4 +(i/4.0), 2.0};           
        }        
    }

    //FermionMatrixHubbardExp Object M for phi and ring lattice
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<T>> M(&ring,phi,0.1);

    //summing up 
    for(int k=0; k<size0; k++){      
        sum = sum + phi(k,0);        
    }

    //phisum matrix 
    for(i=0; i<size1; i++){
        phisum(0,i) = sum*I;
    }
    phisum.exp();

    //logdetM() = logdet(1 + exp(/Phi)exp_hopping_matrix(Kappa_tilda * Nt)) where /Phi = sum over all time slices
    NSL::complex<T> result = NSL::LinAlg::logdet(NSL::LinAlg::Matrix::Identity(Id,size1) + (((phisum) * ring.exp_hopping_matrix(delta*size0)))); //ring.exp_hopping_matrix(delta*size0));
    
    
    //TEST

    //double ratio = fabs(1 - (result.real()/M.logDetM().real()));
    T res1 = fabs(M.logDetM().real() - result.real());
    T res2 = fabs(M.logDetM().imag() - result.imag());
    
    //REQUIRE(ratio <= limit);
    REQUIRE(res1 <= limit);
    REQUIRE(res2 <= limit);
    
}


//Test cases
REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: M", "[fermionMatrixHubbardExp, M]" ) {

    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);

    test_fermionMatrixHubbardExp_M<TestType>(size_0, size_1);

}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: Mdagger", "[fermionMatrixHubbardExp, Mdagger]" ) {

    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);

    test_fermionMatrixHubbardExp_Mdagger<TestType>(size_0, size_1);

}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: MMdagger", "[fermionMatrixHubbardExp, MMdagger]" ) {

    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);

    test_fermionMatrixHubbardExp_MMdagger<TestType>(size_0, size_1);

}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: MdaggerM", "[fermionMatrixHubbardExp, MdaggerM]" ) {

    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);

    test_fermionMatrixHubbardExp_MdaggerM<TestType>(size_0, size_1);

}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_1", "[fermionMatrixHubbardExp, logDetM_1]" ) {
    
    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM_1<TestType>(size_0, size_1);
    
}   
REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_2", "[fermionMatrixHubbardExp, logDetM_2]" ) {
    
    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM_2<TestType>(size_0, size_1);
    
}
REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_3", "[fermionMatrixHubbardExp, logDetM_3]" ) {
    
    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM_3<TestType>(size_0, size_1);
    
}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_4", "[fermionMatrixHubbardExp, logDetM_4]" ) {
    
    const size_type size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const size_type size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM_4<TestType>(size_0, size_1);

}



