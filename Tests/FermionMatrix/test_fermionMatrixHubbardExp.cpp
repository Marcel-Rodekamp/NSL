#include "../test.hpp"

template<NSL::Concept::isNumber Type>
void test_fermionMatrixHubbardExp_M(const NSL::size_t nt, const NSL::size_t nx, const Type & beta = 2);

template<NSL::Concept::isNumber Type>
void test_fermionMatrixHubbardExp_Mdagger(const NSL::size_t nt, const NSL::size_t nx, const Type & beta = 2);

template<NSL::Concept::isNumber Type>
void test_fermionMatrixHubbardExp_MMdagger(const NSL::size_t nt, const NSL::size_t nx, const Type & beta = 2);

template<NSL::Concept::isNumber Type>
void test_fermionMatrixHubbardExp_MdaggerM(const NSL::size_t nt, const NSL::size_t nx, const Type & beta = 2);

template<NSL::Concept::isNumber Type>
void test_logDetM_time_shift_invariance(const NSL::size_t nt, const NSL::size_t nx, const Type & beta = 1);

template<NSL::Concept::isNumber Type>
void test_logDetM_phi_plus_two_pi(const NSL::size_t nt, const NSL::size_t nx, const Type & beta = 1);

template<NSL::Concept::isNumber Type>
void test_logDetM_noninteracting(const NSL::size_t nt, const NSL::size_t nx, const Type & beta = 1);

template<NSL::Concept::isNumber Type>
void test_logDetM_uniform_timeslices(const NSL::size_t nt, const NSL::size_t nx, const Type & beta = 1);

//! The logDetM_* tests are expected to fail, consider issue #36 & #43 

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: M", "[fermionMatrixHubbardExp, M]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    test_fermionMatrixHubbardExp_M<TestType>(nt, nx);

}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: Mdagger", "[fermionMatrixHubbardExp, Mdagger]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    test_fermionMatrixHubbardExp_Mdagger<TestType>(nt, nx);

}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: MMdagger", "[fermionMatrixHubbardExp, MMdagger]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    test_fermionMatrixHubbardExp_MMdagger<TestType>(nt, nx);

}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: MdaggerM", "[fermionMatrixHubbardExp, MdaggerM]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    test_fermionMatrixHubbardExp_MdaggerM<TestType>(nt, nx);

}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_time_shift_invariance", "[fermionMatrixHubbardExp, logDetM_time_shift_invariance]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM_time_shift_invariance<TestType>(nt, nx);
    
}   
REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_phi_plus_two_pi", "[fermionMatrixHubbardExp, logDetM_phi_plus_two_pi]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM_phi_plus_two_pi<TestType>(nt, nx);
    
}
REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_noninteracting", "[fermionMatrixHubbardExp, logDetM_noninteracting]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM_noninteracting<TestType>(nt, nx);
    
}

REAL_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_uniform_timeslices", "[fermionMatrixHubbardExp, logDetM_uniform_timeslices]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logDetM_uniform_timeslices<TestType>(nt, nx);
}

// ======================================================================
// Implementation Details: fermionMatrixHubbardExp_M
// ======================================================================

//Test for the function M(psi)
template<NSL::Concept::isNumber Type>
void test_fermionMatrixHubbardExp_M(const NSL::size_t nt, const NSL::size_t nx, const Type & beta) {

    //hardcoding the calculation done in the method M of fermionMatrixHubbardExp class
    NSL::Tensor<NSL::complex<Type>> phi(nt, nx);
    NSL::Tensor<NSL::complex<Type>> psi(nt, nx);     
    psi.rand();

    NSL::Lattice::Ring<NSL::complex<Type>> r(nx);
    double delta = beta/nt;
    NSL::FermionMatrix::HubbardExp M(r,phi,beta);
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
void test_fermionMatrixHubbardExp_Mdagger(const NSL::size_t nt, const NSL::size_t nx, const Type & beta) {

    //hardcoding the calculation done in the method Mdagger of fermionMatrixHubbardExp class
    NSL::Tensor<NSL::complex<Type>> phi(nt, nx);
    NSL::Tensor<NSL::complex<Type>> psi(nt, nx);     
    psi.rand();

    NSL::Lattice::Ring<NSL::complex<Type>> r(nx);
    double delta = beta/nt;
    NSL::FermionMatrix::HubbardExp M(r,phi,beta);
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
void test_fermionMatrixHubbardExp_MMdagger(const NSL::size_t nt, const NSL::size_t nx, const Type & beta) {

    //hardcoding the calculation done in the method MMdagger of fermionMatrixHubbardExp class
    NSL::Tensor<NSL::complex<Type>> phi(nt, nx);
    NSL::Tensor<NSL::complex<Type>> psi(nt, nx);     
    psi.rand();


    NSL::Lattice::Ring<NSL::complex<Type>> r(nx);
    double delta = beta/nt;
    NSL::FermionMatrix::HubbardExp M(r,phi,beta);
    
    // apply kronecker delta
    NSL::Tensor<NSL::complex<Type>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::Tensor<NSL::complex<Type>> out =  M.M(psi) 
                + M.Mdagger(psi)
                + NSL::LinAlg::mat_vec(
                    (r.exp_hopping_matrix(2*delta)), 
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
void test_fermionMatrixHubbardExp_MdaggerM(const NSL::size_t nt, const NSL::size_t nx, const Type & beta) {

    //hardcoding the calculation done in the method MdaggerM of fermionMatrixHubbardExp class
    NSL::Tensor<NSL::complex<Type>> phi(nt, nx);
    NSL::Tensor<NSL::complex<Type>> psi(nt, nx);     
    psi.rand();

    NSL::Lattice::Ring<NSL::complex<Type>> r(nx);
    double delta = beta/nt;
    NSL::FermionMatrix::HubbardExp M(r,phi,beta);
    NSL::complex<Type> min_I ={0,-1}, I={0,1};
    
    // apply kronecker delta
    NSL::Tensor<NSL::complex<Type>> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::Tensor<NSL::complex<Type>> out =  M.M(psi) + M.Mdagger(psi)+ ((((phi*min_I).exp()).transpose())  *
        NSL::LinAlg::mat_vec((r.exp_hopping_matrix(2*delta)), NSL::LinAlg::transpose((phi*I).exp()*psi))).
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
// Implementation Details: test_logDetM_time_shift_invariance
// ======================================================================

//Test for the function logDetM() (shift in phi)
template<NSL::Concept::isNumber Type>
void test_logDetM_time_shift_invariance(const NSL::size_t nt, const NSL::size_t nx, const Type & beta) {

    int slices_to_shift_by=4;
    NSL::Tensor<NSL::complex<Type>> phi(nt, nx), phiShift(nt, nx);
    phi.rand();
    NSL::Lattice::Ring<NSL::complex<Type>> ring(nx);
    double delta = beta/nt;
    
    //FermionMatrixHubbardExp Object M for ring lattice 
    NSL::FermionMatrix::HubbardExp M(ring,phi,beta);
    
    //FermionMatrixHubbardExp Object Mshift for the shifted phi
    phiShift=NSL::LinAlg::shift(phi,slices_to_shift_by);
    NSL::FermionMatrix::HubbardExp Mshift(ring,phiShift,beta);
    
    NSL::complex<Type> result_shift = Mshift.logDetM();
    NSL::complex<Type> result = M.logDetM();

    INFO("result unshifted: "+std::to_string(result.real())+"+i*"+std::to_string(result.imag()));
    INFO("result   shifted: "+std::to_string(result_shift.real())+"+i*"+std::to_string(result_shift.imag()));
    
    REQUIRE(almost_equal(result_shift,result));
}

// ======================================================================
// Implementation Details: test_logDetM_phi_plus_two_pi
// ======================================================================


//Test for logDetM() (adding 2*pi in one of the time slices )
template<NSL::Concept::isNumber Type>
void test_logDetM_phi_plus_two_pi(const NSL::size_t nt, const NSL::size_t nx, const Type & beta) {

    NSL::Tensor<NSL::complex<Type>> phi(nt, nx), phiShift(nt, nx);
    phi.rand();
    NSL::Lattice::Ring<NSL::complex<Type>> ring(nx);
    Type delta = beta/nt;
    
    NSL::FermionMatrix::HubbardExp M(ring,phi,beta);
    
    //generating random time slice to add 2*pi
    srand (time(NULL));
    int t = rand() % nt;
    //pi
    NSL::complex<Type> pi ={2*std::numbers::pi,0};
    //adding 2*pi at the t_th position
    for(int i=0; i< nx; i++){
        phi(t, i)=phi(t,i) + pi;
        }
    //FermionMatrixHubbardExp Object with modified phi    
    NSL::FermionMatrix::HubbardExp Mshift(ring,phi,beta);
    
    NSL::complex<Type> result_shift = Mshift.logDetM();
    NSL::complex<Type> result = M.logDetM();

    INFO("result unshifted: "+std::to_string(result.real())+"+i*"+std::to_string(result.imag()));
    INFO("result   shifted: "+std::to_string(result_shift.real())+"+i*"+std::to_string(result_shift.imag()));

    REQUIRE(almost_equal(result_shift,result));

}

// ======================================================================
// Implementation Details: test_logDetM_noninteracting
// ======================================================================

//Test for logDetM() when phi=0
template<NSL::Concept::isNumber Type>
void test_logDetM_noninteracting(const NSL::size_t nt, const NSL::size_t nx, const Type & beta) {

    NSL::Tensor<NSL::complex<Type>> phi(nt, nx);
    NSL::Lattice::Ring<NSL::complex<Type>> ring(nx);
    Type delta = beta/nt;
    NSL::FermionMatrix::HubbardExp M(ring,phi,beta);
    
    //When phi=0, logDetM = logdet(1 + exp_hopping_matrix(beta))
    NSL::complex<Type> result_exa = NSL::LinAlg::logdet(
            NSL::Matrix::Identity<Type>(nx) + ring.exp_hopping_matrix(beta)
    );
    
    NSL::complex<Type> result_alg = M.logDetM();
    
    INFO("result algorithm: "+std::to_string(result_alg.real())+"+i*"+std::to_string(result_alg.imag()));
    INFO("result exact    : "+std::to_string(result_exa.real())+"+i*"+std::to_string(result_exa.imag()));

    REQUIRE(almost_equal(result_alg,result_exa));
    
}

// ======================================================================
// Implementation Details: test_logDetM_uniform_timeslices
// ======================================================================

//Test for logDetM() when  all the elemets in every time slice are same
template<NSL::Concept::isNumber Type>
void test_logDetM_uniform_timeslices(const NSL::size_t nt, const NSL::size_t nx, const Type & beta) {

    NSL::Tensor<NSL::complex<Type>> phi(nt, nx), phisum(1,nx);    
    NSL::Lattice::Ring<NSL::complex<Type>> ring(nx);
    Type delta = beta/nt;
    NSL::complex<Type> I ={0,1};

    //setting up phi such that all the elements in a time slice are same
    int i=0, j=0;
    NSL::complex<Type> sum ={0,0};
    for(i=0; i<nt; i++){
        for(j=0; j<nx; j++){
            phi(i,j) = {(2.0*i) +0.4 +(i/4.0), 2.0};           
        }        
    }

    //FermionMatrixHubbardExp Object M for phi and ring lattice
    NSL::FermionMatrix::HubbardExp M(ring,phi,beta);

    //summing up 
    for(int k=0; k<nt; k++){      
        sum = sum + phi(k,0);        
    }

    //phisum matrix 
    for(i=0; i<nx; i++){
        phisum(0,i) = sum*I;
    }
    phisum.exp();

    //logdetM() = logdet(1 + exp(/Phi)exp_hopping_matrix(Kappa_tilda * Nt)) where /Phi = sum over all time slices
    NSL::complex<Type> result_exa = NSL::LinAlg::logdet(
            NSL::Matrix::Identity<NSL::complex<Type>>(nx) + 
            (((phisum) * ring.exp_hopping_matrix(beta)))
    );
    
    
    NSL::complex<Type> result_alg = M.logDetM();

    INFO("result algorithm: "+std::to_string(result_alg.real())+"+i*"+std::to_string(result_alg.imag()));
    INFO("result exact    : "+std::to_string(result_exa.real())+"+i*"+std::to_string(result_exa.imag()));

    REQUIRE(almost_equal(result_alg,result_exa));
    
}


//Test cases

