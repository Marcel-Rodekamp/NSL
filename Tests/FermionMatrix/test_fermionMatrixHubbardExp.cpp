#include "../test.hpp"

template<NSL::Concept::isNumber Type>
void test_fermionMatrixHubbardExp_M(const NSL::size_t size0, const NSL::size_t size1);

template<NSL::Concept::isNumber Type>
void test_fermionMatrixHubbardExp_Mdagger(const NSL::size_t size0, const NSL::size_t size1);

template<NSL::Concept::isNumber Type>
void test_fermionMatrixHubbardExp_MMdagger(const NSL::size_t size0, const NSL::size_t size1);

template<NSL::Concept::isNumber Type>
void test_fermionMatrixHubbardExp_MdaggerM(const NSL::size_t size0, const NSL::size_t size1);

template<NSL::Concept::isNumber Type>
void test_logDetM_1(const NSL::size_t size0, const NSL::size_t size1);

template<NSL::Concept::isNumber Type>
void test_logDetM_2(const NSL::size_t size0, const NSL::size_t size1);

template<NSL::Concept::isNumber Type>
void test_logDetM_3(const NSL::size_t size0, const NSL::size_t size1);

template<NSL::Concept::isNumber Type>
void test_logDetM_4(const NSL::size_t size0, const NSL::size_t size1);

//! The logDetM_* tests are expected to fail, considere issue #36 & #43 

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: M", "[fermionMatrixHubbardExp, M]" ) {
    const NSL::size_t size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);

    test_fermionMatrixHubbardExp_M<TestType>(size_0, size_1);

}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: Mdagger", "[fermionMatrixHubbardExp, Mdagger]" ) {
    const NSL::size_t size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);

    test_fermionMatrixHubbardExp_Mdagger<TestType>(size_0, size_1);

}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: MMdagger", "[fermionMatrixHubbardExp, MMdagger]" ) {
    const NSL::size_t size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);

    test_fermionMatrixHubbardExp_MMdagger<TestType>(size_0, size_1);

}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: MdaggerM", "[fermionMatrixHubbardExp, MdaggerM]" ) {
    const NSL::size_t size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);

    test_fermionMatrixHubbardExp_MdaggerM<TestType>(size_0, size_1);

}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_1", "[fermionMatrixHubbardExp, logDetM_1]" ) {
    const NSL::size_t size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM_1<TestType>(size_0, size_1);
    
}   
REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_2", "[fermionMatrixHubbardExp, logDetM_2]" ) {
    const NSL::size_t size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM_2<TestType>(size_0, size_1);
    
}
REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_3", "[fermionMatrixHubbardExp, logDetM_3]" ) {
    const NSL::size_t size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM_3<TestType>(size_0, size_1);
    
}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_4", "[fermionMatrixHubbardExp, logDetM_4]" ) {
    const NSL::size_t size_0 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t size_1 = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM_4<TestType>(size_0, size_1);
}

// ======================================================================
// Implementation Details: fermionMatrixHubbardExp_M
// ======================================================================

//Test for the function M(psi)
template<NSL::Concept::isNumber Type>
void test_fermionMatrixHubbardExp_M(const NSL::size_t size0, const NSL::size_t size1) {

    //hardcoding the calculation done in the method M of fermionMatrixHubbardExp class
    NSL::Tensor<NSL::complex<Type>> phi(size0, size1);
    NSL::Tensor<NSL::complex<Type>> psi(size0, size1);     
    psi.rand();

    NSL::Lattice::Ring<Type> r(size1);
    //delta= beta/Nt
    double delta = 2.0/size0;
    //FermionMatrixHubbardExp Object M
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi,2.0);
    NSL::complex<Type> I ={0,1};

    // apply kronecker delta
    NSL::Tensor<NSL::complex<Type>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::Tensor<NSL::complex<Type>> out = NSL::LinAlg::mat_vec(       
        r.exp_hopping_matrix(/*delta=(beta/Nt) */delta),
        ((phi*I).exp() * psiShift).transpose()
    );

    // anti-periodic boundary condition
    out.transpose();
    out.slice(0,0,1)*=-1;
    NSL::Tensor<NSL::complex<Type>> result_exa = psi - out;
    NSL::Tensor<NSL::complex<Type>> result_alg = M.M(psi);

    REQUIRE(result_exa.dim() == result_alg.dim());
    REQUIRE(result_exa.numel() == result_alg.numel());
    for(int d = 0; d < result_exa.dim(); ++d){
        REQUIRE(result_exa.shape(d) == result_alg.shape(d));
    }

    for(int i = 0; i < result_exa.numel(); ++i){
        REQUIRE(almost_equal(result_exa[i],result_alg[i]));
    }
}

// ======================================================================
// Implementation Details: fermionMatrixHubbardExp_Mdagger
// ======================================================================

//Test for the function Mdagger(psi)
template<NSL::Concept::isNumber Type>
void test_fermionMatrixHubbardExp_Mdagger(const NSL::size_t size0, const NSL::size_t size1) {

    //hardcoding the calculation done in the method Mdagger of fermionMatrixHubbardExp class
    NSL::Tensor<NSL::complex<Type>> phi(size0, size1);
    NSL::Tensor<NSL::complex<Type>> psi(size0, size1);     
    psi.rand();

    NSL::Lattice::Ring<Type> r(size1);
    //delta=beta/Nt
    double delta = 2.0/size0;
    //FermionMatrixHubbardExp Object M
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi,2.0);
    NSL::complex<Type> min_I ={0,-1};

    // apply kronecker delta
    NSL::Tensor<NSL::complex<Type>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::Tensor<NSL::complex<Type>> out =  (NSL::LinAlg::mat_vec(
        ((phi*min_I).exp()),
        r.exp_hopping_matrix(/*delta=(beta/Nt) */delta)
        
    )) * psiShift;

    // anti-periodic boundary condition
    out.slice(0,0,1)*=-1;
    NSL::Tensor<NSL::complex<Type>> result_exa = psi - out;
    NSL::Tensor<NSL::complex<Type>> result_alg = M.Mdagger(psi);

    //TEST  
    REQUIRE(result_exa.dim() == result_alg.dim());
    REQUIRE(result_exa.numel() == result_alg.numel());
    for(int d = 0; d < result_exa.dim(); ++d){
        REQUIRE(result_exa.shape(d) == result_alg.shape(d));
    }

    for(int i = 0; i < result_exa.numel(); ++i){
        REQUIRE(almost_equal(result_exa[i],result_alg[i]));
    }
}

// ======================================================================
// Implementation Details: fermionMatrixHubbardExp_MMdagger
// ======================================================================

//Test for the function MMdagger(psi)
template<NSL::Concept::isNumber Type>
void test_fermionMatrixHubbardExp_MMdagger(const NSL::size_t size0, const NSL::size_t size1) {

    //hardcoding the calculation done in the method MMdagger of fermionMatrixHubbardExp class
    NSL::Tensor<NSL::complex<Type>> phi(size0, size1);
    NSL::Tensor<NSL::complex<Type>> psi(size0, size1);     
    psi.rand();


    NSL::Lattice::Ring<Type> r(size1);
    //delta=beta/Nt
    double delta = 2.0/size0;
    //FermionMatrixHubbardExp Object M
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi,2.0);
    
    // apply kronecker delta
    NSL::Tensor<NSL::complex<Type>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::Tensor<NSL::complex<Type>> out =  M.M(psi) 
                + M.Mdagger(psi)
                + NSL::LinAlg::mat_vec(
                    (r.exp_hopping_matrix(delta))*(r.exp_hopping_matrix(delta)), 
                    NSL::LinAlg::transpose(psi)
                ).transpose();

    NSL::Tensor<NSL::complex<Type>> result_exa = out - psi;

    NSL::Tensor<NSL::complex<Type>> result_alg = M.MMdagger(psi);

    //TEST  
    REQUIRE(result_exa.dim() == result_alg.dim());
    REQUIRE(result_exa.numel() == result_alg.numel());
    for(int d = 0; d < result_exa.dim(); ++d){
        REQUIRE(result_exa.shape(d) == result_alg.shape(d));
    }

    for(int i = 0; i < result_exa.numel(); ++i){
        REQUIRE(almost_equal(result_exa[i],result_alg[i]));
    }
}


// ======================================================================
// Implementation Details: fermionMatrixHubbardExp_MdaggerM
// ======================================================================

//Test for the function MdaggerM(psi)
template<NSL::Concept::isNumber Type>
void test_fermionMatrixHubbardExp_MdaggerM(const NSL::size_t size0, const NSL::size_t size1) {

    //hardcoding the calculation done in the method MdaggerM of fermionMatrixHubbardExp class
    NSL::Tensor<NSL::complex<Type>> phi(size0, size1);
    NSL::Tensor<NSL::complex<Type>> psi(size0, size1);     
    psi.rand();

    NSL::Lattice::Ring<Type> r(size1);
    //delta=beta/Nt
    double delta = 2.0/size0;
    //FermionMatrixHubbardExp Object M
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r,phi,2.0);
    NSL::complex<Type> min_I ={0,-1}, I={0,1};
    
    // apply kronecker delta
    NSL::Tensor<NSL::complex<Type>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::Tensor<NSL::complex<Type>> out =  M.M(psi) + M.Mdagger(psi)+ ((((phi*min_I).exp()).transpose())  *
        NSL::LinAlg::mat_vec((r.exp_hopping_matrix(delta))*
        (r.exp_hopping_matrix(delta)), NSL::LinAlg::transpose((phi*I).exp()*psi))).
    transpose();

    NSL::Tensor<NSL::complex<Type>> result_exa = out - psi;

    NSL::Tensor<NSL::complex<Type>> result_alg = M.MdaggerM(psi);

    //TEST  
    REQUIRE(result_exa.dim() == result_alg.dim());
    REQUIRE(result_exa.numel() == result_alg.numel());
    for(int d = 0; d < result_exa.dim(); ++d){
        REQUIRE(result_exa.shape(d) == result_alg.shape(d));
    }

    for(int i = 0; i < result_exa.numel(); ++i){
        REQUIRE(almost_equal(result_exa[i],result_alg[i]));
    }
}

// ======================================================================
// Implementation Details: test_logDetM_1
// ======================================================================

//Test for the function logDetM() (shift in phi)
template<NSL::Concept::isNumber Type>
void test_logDetM_1(const NSL::size_t size0, const NSL::size_t size1) {

    NSL::Tensor<NSL::complex<Type>> phi(size0, size1), phiShift(size0, size1);
    phi.rand();
    NSL::Lattice::Ring<Type> ring(size1);
    //delta=beta/Nt
    double delta = 0.1/size0;
    
    //FermionMatrixHubbardExp Object M for ring lattice 
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<Type>> M(&ring,phi,0.1);
    
    //FermionMatrixHubbardExp Object Mshift for the shifted phi
    phiShift=NSL::LinAlg::shift(phi,4);
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<Type>> Mshift(&ring,phiShift,0.1);
    
    NSL::complex<Type> result_shift = Mshift.logDetM();
    NSL::complex<Type> result = M.logDetM();

    INFO("result unshifted: "+std::to_string(result.real())+"+i*"+std::to_string(result.imag()));
    INFO("result   shifted: "+std::to_string(result_shift.real())+"+i*"+std::to_string(result_shift.imag()));
    
    REQUIRE(almost_equal(result_shift,result));
}

// ======================================================================
// Implementation Details: test_logDetM_2
// ======================================================================


//Test for logDetM() (adding 2*pi in one of the time slices )
template<NSL::Concept::isNumber Type>
void test_logDetM_2(const NSL::size_t size0, const NSL::size_t size1) {

    NSL::Tensor<NSL::complex<Type>> phi(size0, size1), phiShift(size0, size1);
    phi.rand();
    NSL::Lattice::Ring<Type> ring(size1);
    Type delta = 0.1/size0;
    
    //FermionMatrixHubbardExp Object M 
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<Type>> M(&ring,phi,0.1);
    
    //generating random time slice to add 2*pi
    srand (time(NULL));
    int t = rand() % size0;
    //pi
    NSL::complex<Type> pi ={2*std::numbers::pi,0};
    //adding 2*pi at the t_th position
    for(int i=0; i< size1; i++){
        phi(t, i)=phi(t,i) + pi;
        }
    //FermionMatrixHubbardExp Object with modified phi    
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<Type>> Mshift(&ring,phi,0.1);
    
    NSL::complex<Type> result_shift = Mshift.logDetM();
    NSL::complex<Type> result = M.logDetM();

    INFO("result unshifted: "+std::to_string(result.real())+"+i*"+std::to_string(result.imag()));
    INFO("result   shifted: "+std::to_string(result_shift.real())+"+i*"+std::to_string(result_shift.imag()));

    REQUIRE(almost_equal(result_shift,result));

}

// ======================================================================
// Implementation Details: test_logDetM_3
// ======================================================================

//Test for logDetM() when phi=0
template<NSL::Concept::isNumber Type>
void test_logDetM_3(const NSL::size_t size0, const NSL::size_t size1) {

    NSL::Tensor<NSL::complex<Type>> phi(size0, size1); // phiShift(size0, size1);    
    NSL::Lattice::Ring<Type> ring(size1);
    //delta=beta/Nt
    Type delta = 1./size0;
    //FermionMatrixHubbardExp Object 
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&ring,phi,1.);
    
    //When phi=0, logDetM = logdet(1 + exp_hopping_matrix(beta))
    NSL::complex<Type> result_exa = NSL::LinAlg::logdet(
            NSL::Matrix::Identity<Type>(size1) + ring.exp_hopping_matrix(1.)
    );
    
    NSL::complex<Type> result_alg = M.logDetM();
    
    INFO("result algorithm: "+std::to_string(result_alg.real())+"+i*"+std::to_string(result_alg.imag()));
    INFO("result exact    : "+std::to_string(result_exa.real())+"+i*"+std::to_string(result_exa.imag()));

    REQUIRE(almost_equal(result_alg,result_exa));
    
}

// ======================================================================
// Implementation Details: test_logDetM_4
// ======================================================================

//Test for logDetM() when  all the elemets in every time slice are same
template<NSL::Concept::isNumber Type>
void test_logDetM_4(const NSL::size_t size0, const NSL::size_t size1) {

    NSL::Tensor<NSL::complex<Type>> phi(size0, size1), phisum(1,size1);    
    NSL::Lattice::Ring<Type> ring(size1);
    //delta=beta/Nt
    Type delta = 1./size0;
    NSL::complex<Type> I ={0,1};

    //setting up phi such that all the elements in a time slice are same
    int i=0, j=0;
    NSL::complex<Type> sum ={0,0};
    for(i=0; i<size0; i++){
        for(j=0; j<size1; j++){
            phi(i,j) = {(2.0*i) +0.4 +(i/4.0), 2.0};           
        }        
    }

    //FermionMatrixHubbardExp Object M for phi and ring lattice
    NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<Type>> M(&ring,phi,1.);

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
    NSL::complex<Type> result_exa = NSL::LinAlg::logdet(
            NSL::Matrix::Identity<NSL::complex<Type>>(size1) + 
            (((phisum) * ring.exp_hopping_matrix(delta*size0)))
    ); //ring.exp_hopping_matrix(delta*size0));
    
    
    NSL::complex<Type> result_alg = M.logDetM();

    INFO("result algorithm: "+std::to_string(result_alg.real())+"+i*"+std::to_string(result_alg.imag()));
    INFO("result exact    : "+std::to_string(result_exa.real())+"+i*"+std::to_string(result_exa.imag()));

    REQUIRE(almost_equal(result_alg,result_exa));
    
}


//Test cases

