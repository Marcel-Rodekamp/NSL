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
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_logDetM_time_shift_invariance<TestType>(nt, Lattice);

}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardDiag: logDetM_phi_plus_two_pi", "[fermionMatrixHubbardDiag, logDetM_phi_plus_two_pi]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_logDetM_phi_plus_two_pi<TestType>(nt, Lattice);

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

    NSL::FermionMatrix::HubbardDiag M(Lattice,phi,beta);
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

    NSL::FermionMatrix::HubbardDiag M(Lattice,phi,beta);
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

    NSL::FermionMatrix::HubbardDiag M(Lattice,phi,beta);
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

    NSL::FermionMatrix::HubbardDiag M(Lattice,phi,beta);
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

    NSL::FermionMatrix::HubbardDiag<Type,LatticeType> M     (Lattice,phi     ,beta);
    NSL::FermionMatrix::HubbardDiag<Type,LatticeType> Mshift(Lattice,phiShift,beta);

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

    NSL::FermionMatrix::HubbardDiag M     (Lattice,phi     ,beta);
    NSL::FermionMatrix::HubbardDiag Mshift(Lattice,phiShift,beta);

    Type result = M.logDetM();
    Type result_shift = Mshift.logDetM();

    INFO("nx: "+std::to_string(nx)+" nt: "+std::to_string(nt));
    INFO("result unshifted: "+NSL::to_string(result));
    INFO("result   shifted: "+NSL::to_string(result_shift));
    INFO("difference      : "+NSL::to_string(result-result_shift));

    //comparing only the real parts
    REQUIRE(almost_equal(result_shift.real(),result.real(),std::numeric_limits<Type>::digits10-1));

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
    NSL::FermionMatrix::HubbardDiag M(Lattice,phi,beta);
    
    //When phi=0, logDetM = logdet(1 + exp_hopping_matrix(beta))
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