#include "../test.hpp"

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_M(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_Mdagger(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MMdagger(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MMdagger_consistency(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MdaggerM(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MdaggerM_consistency(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_logDetM_time_shift_invariance(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 1);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_logDetM_phi_plus_two_pi(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 1);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_logDetM_noninteracting(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 1);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_logDetM_uniform_timeslices(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 1);

//! The logDetM_* tests are expected to fail, consider issue #36 & #43 

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardExp: M", "[fermionMatrixHubbardExp, M]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_fermionMatrixHubbardExp_M<TestType>(nt, Lattice);

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardExp: Mdagger", "[fermionMatrixHubbardExp, Mdagger]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_fermionMatrixHubbardExp_Mdagger<TestType>(nt, Lattice);

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardExp: MMdagger", "[fermionMatrixHubbardExp, MMdagger]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_fermionMatrixHubbardExp_MMdagger<TestType>(nt, Lattice);

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardExp: MMdagger consistency", "[fermionMatrixHubbardExp, MMdagger, consistency]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_fermionMatrixHubbardExp_MMdagger_consistency<TestType>(nt, Lattice);

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardExp: MdaggerM", "[fermionMatrixHubbardExp, MdaggerM]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_fermionMatrixHubbardExp_MdaggerM<TestType>(nt, Lattice);

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardExp: MdaggerM consistency", "[fermionMatrixHubbardExp, MdaggerM, consistency]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_fermionMatrixHubbardExp_MdaggerM_consistency<TestType>(nt, Lattice);

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_time_shift_invariance", "[fermionMatrixHubbardExp, logDetM_time_shift_invariance]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_logDetM_time_shift_invariance<TestType>(nt, Lattice);

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_phi_plus_two_pi", "[fermionMatrixHubbardExp, logDetM_phi_plus_two_pi]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_logDetM_phi_plus_two_pi<TestType>(nt, Lattice);

}
COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_noninteracting", "[fermionMatrixHubbardExp, logDetM_noninteracting]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_logDetM_noninteracting<TestType>(nt, Lattice);

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardExp: logDetM_uniform_timeslices", "[fermionMatrixHubbardExp, logDetM_uniform_timeslices]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_logDetM_uniform_timeslices<TestType>(nt, Lattice);

}

// ======================================================================
// Implementation Details: fermionMatrixHubbardExp_M
// ======================================================================

//Test for the function M(psi)
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_M(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {
    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();
    //hardcoding the calculation done in the method M of fermionMatrixHubbardExp class
    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(nt, nx);
    phi.rand();
    psi.rand();

    Type delta = beta/nt;
    NSL::FermionMatrix::HubbardExp M(Lattice,phi,beta);
    ComplexType I ={0,1};

    // apply kronecker delta
    NSL::Tensor<Type> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::Tensor<Type> out = NSL::LinAlg::mat_vec(
        Lattice.exp_hopping_matrix(delta),
        ((phi*I).exp() * psiShift).transpose()
    );

    // anti-periodic boundary condition
    out.transpose();
    out.slice(0,0,1)*=-1;
    NSL::Tensor<Type> result_exa = psi - out;
    NSL::Tensor<Type> result_alg = M.M(psi);

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
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_Mdagger(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    //hardcoding the calculation done in the method Mdagger of fermionMatrixHubbardExp class
    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(nt, nx);
    phi.rand();
    psi.rand();

    Type delta = beta/nt;
    NSL::FermionMatrix::HubbardExp M(Lattice,phi,beta);
    ComplexType  I ={0,1};

    // apply kronecker delta
    NSL::Tensor<Type> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::Tensor<Type> out =  (NSL::LinAlg::mat_vec(
        ((-1*phi*I).exp()),
        Lattice.exp_hopping_matrix(/*delta=(beta/Nt) */delta)
        
    )) * psiShift;

    // anti-periodic boundary condition
    out.slice(0,0,1)*=-1;
    NSL::Tensor<Type> result_exa = psi - out;
    NSL::Tensor<Type> result_alg = M.Mdagger(psi);

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
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MMdagger(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    //hardcoding the calculation done in the method MMdagger of fermionMatrixHubbardExp class
    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(nt, nx);
    phi.rand();
    psi.rand();

    Type delta = beta/nt;
    NSL::FermionMatrix::HubbardExp M(Lattice,phi,beta);
    
    // apply kronecker delta
    NSL::Tensor<Type> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::Tensor<Type> out =  M.M(psi) 
                + M.Mdagger(psi)
                + NSL::LinAlg::mat_vec(
                    (Lattice.exp_hopping_matrix(2*delta)), 
                    NSL::LinAlg::transpose(psi)
                ).transpose();

    NSL::Tensor<Type> result_exa = out - psi;

    NSL::Tensor<Type> result_alg = M.MMdagger(psi);

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
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MdaggerM(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    //hardcoding the calculation done in the method MdaggerM of fermionMatrixHubbardExp class
    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(nt, nx);
    phi.rand();
    psi.rand();

    Type delta = beta/nt;
    NSL::FermionMatrix::HubbardExp M(Lattice,phi,beta);
    ComplexType I={0,1};
    
    // apply kronecker delta
    NSL::Tensor<Type> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::Tensor<Type> out =  M.M(psi) + M.Mdagger(psi)+ ((((-1*phi*I).exp()).transpose())  *
        NSL::LinAlg::mat_vec((Lattice.exp_hopping_matrix(2*delta)), NSL::LinAlg::transpose((phi*I).exp()*psi))).
    transpose();

    NSL::Tensor<Type> result_exa = out - psi;

    NSL::Tensor<Type> result_alg = M.MdaggerM(psi);

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
// Implementation Details: test_MdaggerM_consistency
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MdaggerM_consistency(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    //hardcoding the calculation done in the method MdaggerM of fermionMatrixHubbardExp class
    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(nt, nx);
    phi.rand();
    psi.rand();

    Type delta = beta/nt;
    NSL::FermionMatrix::HubbardExp M(Lattice,phi,beta);
    ComplexType I={0,1};
 
    auto direct = M.MdaggerM(psi);
    auto indirect = M.Mdagger(M.M(psi));
    auto diff = direct - indirect;

    for(int i = 0; i < direct.numel(); ++i){
        REQUIRE(almost_equal(direct[i],indirect[i]));
    }

}

// ======================================================================
// Implementation Details: test_MdaggerM_consistency
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MMdagger_consistency(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    //hardcoding the calculation done in the method MdaggerM of fermionMatrixHubbardExp class
    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(nt, nx);
    phi.rand();
    psi.rand();

    Type delta = beta/nt;
    NSL::FermionMatrix::HubbardExp M(Lattice,phi,beta);
    ComplexType I={0,1};
 
    auto direct = M.MMdagger(psi);
    auto indirect = M.M(M.Mdagger(psi));
    auto diff = direct - indirect;

    for(int i = 0; i < direct.numel(); ++i){
        REQUIRE(almost_equal(direct[i],indirect[i]));
    }

}

// ======================================================================
// Implementation Details: test_logDetM_time_shift_invariance
// ======================================================================

//Test for the function logDetM() (shift in phi)
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_logDetM_time_shift_invariance(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    // We should find that shifting phi in time doesn't change the determinant.
    int slices_to_shift_by=4;

    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    size_t nx = Lattice.sites();
    NSL::Tensor<Type> phi(nt, nx), phiShift(nt, nx);
    phi.rand();
    phiShift=NSL::LinAlg::shift(phi,slices_to_shift_by);

    Type delta = beta/nt;

    NSL::FermionMatrix::HubbardExp<Type,LatticeType> M     (Lattice,phi     ,beta);
    NSL::FermionMatrix::HubbardExp<Type,LatticeType> Mshift(Lattice,phiShift,beta);

    Type result = M.logDetM();
    Type result_shift = Mshift.logDetM();

    INFO("nx: "+std::to_string(nx)+" nt: "+std::to_string(nt));
    INFO("result unshifted: "+NSL::to_string(result));
    INFO("result   shifted: "+NSL::to_string(result_shift));
    INFO("difference      : "+NSL::to_string(result-result_shift));

    REQUIRE(almost_equal(result_shift,result));
}

// ======================================================================
// Implementation Details: test_logDetM_phi_plus_two_pi
// ======================================================================


//Test for logDetM() (adding 2*pi in one of the time slices )
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_logDetM_phi_plus_two_pi(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    // We should find that by shifting any element of phi by 2π the determinant doesn't change.

    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();
    Type delta = beta/nt;

    NSL::Tensor<Type> phi(nt, nx), phiShift(nt, nx), random(nt, nx);
    phi.rand();

    // We can't generate random integers due to a torch issue.
    // But we can generate Types between 0 and 1 and then truncate to ints.
    random.rand();
    NSL::Tensor<Type> orbits = static_cast<NSL::Tensor<int>>(10*random);

    Type two_pi = 2*std::numbers::pi;
    phiShift = phi + two_pi * orbits;

    NSL::FermionMatrix::HubbardExp M     (Lattice,phi     ,beta);
    NSL::FermionMatrix::HubbardExp Mshift(Lattice,phiShift,beta);

    Type result = M.logDetM();
    Type result_shift = Mshift.logDetM();

    INFO("nx: "+std::to_string(nx)+" nt: "+std::to_string(nt));
    INFO("result unshifted: "+NSL::to_string(result));
    INFO("result   shifted: "+NSL::to_string(result_shift));
    INFO("difference      : "+NSL::to_string(result-result_shift));

    REQUIRE(almost_equal(result_shift,result));

}

// ======================================================================
// Implementation Details: test_logDetM_noninteracting
// ======================================================================

//Test for logDetM() when phi=0
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_logDetM_noninteracting(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    NSL::Tensor<Type> phi(nt, nx);
    Type delta = beta/nt;
    NSL::FermionMatrix::HubbardExp M(Lattice,phi,beta);
    
    //When phi=0, logDetM = logdet(1 + exp_hopping_matrix(beta))
    Type result_exa = NSL::LinAlg::logdet(
            NSL::Matrix::Identity<Type>(nx) + Lattice.exp_hopping_matrix(beta)
    );

    Type result_alg = M.logDetM();

    INFO("nx: "+std::to_string(nx)+" nt: "+std::to_string(nt));
    INFO("result algorithm: "+NSL::to_string(result_alg));
    INFO("result exact    : "+NSL::to_string(result_exa));
    INFO("difference      : "+NSL::to_string(result_exa-result_alg));

    REQUIRE(almost_equal(result_alg,result_exa));
}

// ======================================================================
// Implementation Details: test_logDetM_uniform_timeslices
// ======================================================================

//Test for logDetM() when  all the elements in every time slice are same
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_logDetM_uniform_timeslices(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    NSL::Tensor<Type> phi(nt, nx);
    Type delta = beta/nt;
    ComplexType I ={0,1};

    // When phi on a given timeslice is the same on every spatial site
    for(int i=0; i<nt; i++){
        phi(i, NSL::Slice() ) = 2.0*i + 0.4 + i/4.0;
    }
    // exp(i phi(t)) matrix is proportional to the identity matrix and can be
    // treated like a scalar.
    // When EVERY timeslice is like that we gather all the scalars together
    NSL::Tensor<ComplexType> sum(1);
    sum(0) = I*phi( NSL::Slice(), 0).sum();
    ComplexType expsum = NSL::LinAlg::exp(sum)(0);

    // to get
    //      logdet M = logdet( 1 + exp(sum(phi(t))) exp(kappa_tilde * Nt))
    //               = logdet( 1 + exp(sum(phi(t))) exp(kappa beta)
    NSL::FermionMatrix::HubbardExp M(Lattice,phi,beta);
    Type result_alg = M.logDetM();
    Type result_exa = NSL::LinAlg::logdet(
            NSL::Matrix::Identity<Type>(nx) + 
            expsum * Lattice.exp_hopping_matrix(beta)
    );

    INFO("nx: "+std::to_string(nx)+" nt: "+std::to_string(nt));
    INFO("result algorithm: "+NSL::to_string(result_alg));
    INFO("result exact    : "+NSL::to_string(result_exa));
    INFO("difference      : "+NSL::to_string(result_exa-result_alg));

    REQUIRE(almost_equal(result_alg,result_exa));

}


//Test cases

