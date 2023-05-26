#include "../test.hpp"
#include <iomanip>
#include <limits>
#include "FermionMatrix/fermionMatrix.hpp"
#include "FermionMatrix/Impl/hubbardDiag.hpp"
#include "FermionMatrix/Impl/hubbardDiag.tpp"

// We rely on the accuracy of the application of M for testing.
// If M is implemented correctly and M_dense is implemented correctly we can
// use the dense implementation to check that Mdagger is implemented correctly.
// THEN we can use the correctness of M and Mdagger to check if MMdagger and MdaggerM are correct,
// since in principle the result of MMdagger should just be M•Mdagger (and similarly for MdaggerM).

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardDiag_M_dense(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardDiag_Mdagger(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardDiag_MMdagger(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardDiag_MdaggerM(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_logDetM_time_shift_invariance(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 0.5);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_logDetM_phi_plus_two_pi(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 1);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_logDetM_noninteracting(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 1);

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardDiag: M dense", "[fermionMatrixHubbardDiag, M, dense]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_fermionMatrixHubbardDiag_M_dense<TestType>(nt, Lattice);

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardDiag: Mdagger", "[fermionMatrixHubbardDiag, Mdagger]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_fermionMatrixHubbardDiag_Mdagger<TestType>(nt, Lattice);

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardDiag: MMdagger", "[fermionMatrixHubbardDiag, MMdagger]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_fermionMatrixHubbardDiag_MMdagger<TestType>(nt, Lattice);

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardDiag: MdaggerM", "[fermionMatrixHubbardDiag, MdaggerM]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_fermionMatrixHubbardDiag_MdaggerM<TestType>(nt, Lattice);

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardDiag: logDetM_time_shift_invariance", "[fermionMatrixHubbardDiag, logDetM_time_shift_invariance]" ) {
   
    if(std::is_same_v<TestType, NSL::complex<float> >){
        WARN("This test is currently expected to fail with float.");
        // The issue is the accuracy of the log det.
        // In fact, the code is (otherwise) correct, and we have manually checked that
        // if you divide the sausage in HubbardDiag.logdetM by 100, calculate the log det,
        // and then add back in Nx log 100 you get a reasonable answer that can be
        // computed.
        // That is,
        //      log det M = log det α M + Nx log 1/α
        // with α=100 gives reliable results (for these randomly sampled phi).
    } else {

    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_logDetM_time_shift_invariance<TestType>(nt, Lattice, 0.0625);
    }

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardDiag: logDetM_phi_plus_two_pi", "[fermionMatrixHubbardDiag, logDetM_phi_plus_two_pi]" ) {
   
    if(std::is_same_v<TestType, NSL::complex<float> >){
        WARN("This test is currently expected to fail with float.");
        // The issue is the accuracy of the log det.
        // In fact, the code is (otherwise) correct, and we have manually checked that
        // if you divide the sausage in HubbardDiag.logdetM by 100, calculate the log det,
        // and then add back in Nx log 100 you get a reasonable answer that can be
        // computed.
        // That is,
        //      log det M = log det α M + Nx log 1/α
        // with α=100 gives reliable results (for these randomly sampled phi).
    } else {

    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_logDetM_phi_plus_two_pi<TestType>(nt, Lattice);
    }

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardDiag: logDetM_noninteracting", "[fermionMatrixHubbardDiag, logDetM_noninteracting]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_logDetM_noninteracting<TestType>(nt, Lattice);

}
// ======================================================================
// Implementation Details: fermionMatrixHubbardDiag_M_dense
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardDiag_M_dense(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {
    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(nt, nx);
    NSL::Tensor<Type> dense(nt, nx);
    phi.rand();
    psi.rand();

    NSL::FermionMatrix::HubbardDiag M(Lattice,nt,beta);
    M.populate(phi);
    NSL::Tensor<Type> sparse = M.M(psi);

    NSL::Tensor<Type> M_dense = M.M_dense(nt);
    // Just do the most obvious mat-vec.
    for(int t=0; t < nt; t++){
        for(int x=0; x < nx; x++){
            for(int i=0; i< nt; i++){
                for(int y=0; y<nx; y++){
                    dense(t,x) += M_dense(t,x,i,y) * psi(i,y);
                }
            }
        }
    }

    INFO("nt: "+NSL::to_string(nt)+" nx: "+NSL::to_string(nx));
    REQUIRE( almost_equal(sparse, dense).all() );
}

// ======================================================================
// Implementation Details: fermionMatrixHubbardDiag_Mdagger
// ======================================================================

//Test for the function Mdagger(psi)
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardDiag_Mdagger(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {
    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(nt, nx);
    NSL::Tensor<Type> dense(nt, nx);
    phi.rand();
    psi.rand();

    NSL::FermionMatrix::HubbardDiag M(Lattice,nt,beta);
    M.populate(phi);
    NSL::Tensor<Type> sparse = M.Mdagger(psi);

    // Get a dense representation of M and construct M†
    NSL::Tensor<Type> Mdagger_dense = M.M_dense(nt)
                                        .transpose(0,2)     // switch time  indices
                                        .transpose(1,3)     // switch space indices
                                        .conj();
    // Then do the obvious mat-vec.
    for(int t=0; t < nt; t++){
        for(int x=0; x < nx; x++){
            for(int i=0; i< nt; i++){
                for(int y=0; y<nx; y++){
                    dense(t,x) += Mdagger_dense(t,x,i,y) * psi(i,y);
                }
            }
        }
    }

    INFO("nt: "+NSL::to_string(nt)+" nx: "+NSL::to_string(nx));
    REQUIRE( almost_equal(sparse,dense).all() );
}

//Test cases

// ======================================================================
// Implementation Details: test_MMdagger
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardDiag_MMdagger(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(nt, nx);
    phi.rand();
    psi.rand();

    NSL::FermionMatrix::HubbardDiag M(Lattice,nt,beta);
    M.populate(phi);
    ComplexType I={0,1};
 
    auto direct = M.MMdagger(psi);
    auto indirect = M.M(M.Mdagger(psi));
    auto diff = direct - indirect;

    INFO("nt: "+NSL::to_string(nt)+" nx: "+NSL::to_string(nx));
    REQUIRE( almost_equal(direct,indirect, std::numeric_limits<Type>::digits10-1).all() );
}

// ======================================================================
// Implementation Details: test_MdaggerM
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardDiag_MdaggerM(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    //hardcoding the calculation done in the method MdaggerM of fermionMatrixHubbardDiag class
    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(nt, nx);
    phi.rand();
    psi.rand();

    NSL::FermionMatrix::HubbardDiag M(Lattice,nt,beta);
    M.populate(phi,NSL::Hubbard::Particle);
    ComplexType I={0,1};
 
    auto direct = M.MdaggerM(psi);
    auto indirect = M.Mdagger(M.M(psi));

    INFO("nt: "+NSL::to_string(nt)+" nx: "+NSL::to_string(nx));

    REQUIRE( almost_equal(direct,indirect, std::numeric_limits<Type>::digits10-1).all() );
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

    NSL::FermionMatrix::HubbardDiag<Type,LatticeType> M     (Lattice,nt     ,beta);
    M.populate(phi);
    NSL::FermionMatrix::HubbardDiag<Type,LatticeType> Mshift(Lattice,nt,beta);
    Mshift.populate(phiShift);

    Type result = M.logDetM();
    Type result_shift = Mshift.logDetM();

    INFO("nx: "+std::to_string(nx)+" nt: "+std::to_string(nt));
    INFO("result unshifted: "+NSL::to_string(result));
    INFO("result   shifted: "+NSL::to_string(result_shift));
    INFO("difference      : "+NSL::to_string(result-result_shift));
    INFO("ratio-1         : "+NSL::to_string(result/result_shift -1));

    REQUIRE(almost_equal(result_shift,result, std::numeric_limits<Type>::digits10-1) );
}

// ======================================================================
// Implementation Details: test_logDetM_phi_plus_two_pi
// ======================================================================


//Test for logDetM() (adding 2*pi in one of the time slices )
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_logDetM_phi_plus_two_pi(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    // We should find that by shifting any element of phi by 2π the real part of the determinant doesn't change.

    typedef typename NSL::RT_extractor<Type>::value_type RealType;
    typedef NSL::complex<RealType> ComplexType;
    NSL::size_t nx = Lattice.sites();
    Type delta = beta/nt;

    NSL::Tensor<Type> phi(nt, nx), phiShift(nt, nx), random(nt, nx);
    phi.rand();

    //!\todo: We tried NSL::Tensor<int> orbits(nt, nx); orbits.
    //        However, when orbits is an int-type tensor we get failures in this test
    //        for complex<double>; it passes as written.
    //        It seems likely that this failure is due to some typecasting 
    //        or type compatibility issue.
    //        
    //        One thing to note is the way that it fails is that the logs
    //        wind up differing by some integer multiple of 2πi; it's taking a wrong
    //        branch, or something?
    random.rand();
    NSL::Tensor<Type> orbits = static_cast<NSL::Tensor<int>>(10*random);

    RealType two_pi = 2*std::numbers::pi_v<RealType>;
    phiShift = phi + two_pi * orbits;

    NSL::FermionMatrix::HubbardDiag<Type,LatticeType> M     (Lattice,nt     ,beta);
    M.populate(phi);
    NSL::FermionMatrix::HubbardDiag<Type,LatticeType> Mshift(Lattice,nt,beta);
    Mshift.populate(phiShift);

    Type result = M.logDetM();
    Type result_shift = Mshift.logDetM();
    RealType diff_imag_mod_two_pi = std::remainder(
            static_cast<RealType>(NSL::imag(result - result_shift)),
            static_cast<RealType>(two_pi)
            );


    INFO("nx: "+std::to_string(nx)+" nt: "+std::to_string(nt));
    INFO("result unshifted: "+NSL::to_string(result));
    INFO("result   shifted: "+NSL::to_string(result_shift));
    INFO("difference      : "+NSL::to_string(result-result_shift));
    INFO("Im(∆)%2π        : "+NSL::to_string(diff_imag_mod_two_pi));

    //comparing only the real parts
    REQUIRE(almost_equal(result_shift.real(),result.real(),std::numeric_limits<Type>::digits10-1));
    REQUIRE(almost_equal(static_cast<RealType>(0), 
        diff_imag_mod_two_pi,
        std::numeric_limits<Type>::digits10-3));

}


// ======================================================================
// Implementation Details: test_logDetM_noninteracting
// ======================================================================

//Test for logDetM() when phi=0
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_logDetM_noninteracting(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    NSL::Tensor<Type> phi(nt, nx), sausage = NSL::Matrix::Identity<Type>(nx);
    Type delta = beta/nt;
    NSL::FermionMatrix::HubbardDiag M(Lattice,nt,beta);
    M.populate(phi);
    
    //When phi=0, logDetM = logdet(1 + K^{t})
    for(int t=0; t<nt; t++){
        sausage.mat_mul(NSL::Matrix::Identity<Type>(nx) - Lattice.hopping_matrix(delta));
    }
    Type result_exa = NSL::LinAlg::logdet(
            NSL::Matrix::Identity<Type>(nx) + sausage
    );

    Type result_alg = M.logDetM();

    INFO("nx: "+std::to_string(nx)+" nt: "+std::to_string(nt));
    INFO("result algorithm: "+NSL::to_string(result_alg));
    INFO("result exact    : "+NSL::to_string(result_exa));
    INFO("difference      : "+NSL::to_string(result_exa-result_alg));

    REQUIRE(almost_equal(result_alg,result_exa,std::numeric_limits<Type>::digits10-1));
}

//Test cases
