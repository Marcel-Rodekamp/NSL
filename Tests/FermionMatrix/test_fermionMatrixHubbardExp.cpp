#include "../test.hpp"
#include <iomanip>
#include <limits>

// We rely on the accuracy of the application of M for testing.
// If M is implemented correctly and M_dense is implemented correctly we can
// use the dense implementation to check that Mdagger is implemented correctly.
// THEN we can use the correctness of M and Mdagger to check if MMdagger and MdaggerM are correct,
// since in principle the result of MMdagger should just be M•Mdagger (and similarly for MdaggerM).

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_M(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

// This SHOULD be guaranteed to pass, since it just compares 
//  - (matrix * identity matrix) * vector
//  - matrix * vector
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_M_dense(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_Mdagger(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MMdagger(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MdaggerM(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_M_batched(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_Mdagger_batched(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MMdagger_batched(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MdaggerM_batched(const NSL::size_t nt, LatticeType & Lattice, const Type & beta = 2);


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
    const TestType beta = 2;

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_fermionMatrixHubbardExp_M(nt, Lattice, beta);

    test_fermionMatrixHubbardExp_M_batched(nt, Lattice, beta);
}

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardExp: M dense", "[fermionMatrixHubbardExp, M, dense]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_fermionMatrixHubbardExp_M_dense<TestType>(nt, Lattice);

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

COMPLEX_NSL_TEST_CASE( "fermionMatrixHubbardExp: MdaggerM", "[fermionMatrixHubbardExp, MdaggerM]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 8, 10, 12, 14, 16);
    const NSL::size_t nx = GENERATE(2, 4, 8, 10, 12, 14, 16);

    NSL::Lattice::Ring<TestType> Lattice(nx);
    test_fermionMatrixHubbardExp_MdaggerM<TestType>(nt, Lattice);

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
    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);
    ComplexType I ={0,1};

    // apply kronecker delta
    //NSL::Tensor<Type> psiShift = NSL::LinAlg::shift(psi,1);
    NSL::Tensor<Type> out = NSL::LinAlg::mat_vec(
        Lattice.exp_hopping_matrix(delta),
        ((phi*I).exp() * psi).transpose()
    );

    // anti-periodic boundary condition
    out.transpose();
    out.shift(1,0);

    out.slice(0,0,1)*=-1;
    NSL::Tensor<Type> result_exa = psi - out;
    NSL::Tensor<Type> result_alg = M.M(psi);

    INFO("nt: "+NSL::to_string(nt)+" nx: "+NSL::to_string(nx));
    INFO(fmt::format("result_exa.sum() = {}", NSL::to_string(result_exa.sum())));
    INFO(fmt::format("result_alg.sum() = {}", NSL::to_string(result_alg.sum())));

    REQUIRE( almost_equal(result_exa, result_alg, std::numeric_limits<Type>::digits10).all() );
}

// ======================================================================
// Implementation Details: fermionMatrixHubbardExp_M_dense
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_M_dense(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {
    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(nt, nx);
    NSL::Tensor<Type> dense(nt, nx);
    phi.rand();
    psi.rand();

    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
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
// Implementation Details: fermionMatrixHubbardExp_Mdagger
// ======================================================================

//Test for the function Mdagger(psi)
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_Mdagger(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {
    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();
    INFO("nt: "+NSL::to_string(nt)+" nx: "+NSL::to_string(nx));

    NSL::Tensor<ComplexType> phi(nt, nx);
    NSL::Tensor<ComplexType> psi(nt, nx);
    phi.rand();
    psi.rand();

    // To simplify this test one can force the field
    //phi = phi.real() + ComplexType(0,0);  // phi to be real
    //psi = psi.real() + ComplexType(0,0);  // psi to be real
    //psi = psi.imag() * ComplexType(0,1);  // psi to be imaginary
    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);


    // First let's check Mdagger against the dagger of the dense implementation of M.
    
    // Construct a dense representation of M† from a dense representation of M
    NSL::Tensor<ComplexType> Mdense_dagger = M.M_dense(nt).transpose(0,2).transpose(1,3).conj();

    //  We can also apply Mdagger to the identity matrix in order to get a dense Mdagger.
    //  Follow the M_dense implementation:

    NSL::Tensor<ComplexType> dense(nt, nx, nt, nx);

    // Construct the identity matrix.
    NSL::Tensor<ComplexType> identity(nt, nx, nt, nx);
    for(int t = 0; t < nt; t++){
        identity(t,NSL::Slice(), t, NSL::Slice()) = NSL::Matrix::Identity<ComplexType>(nx);
    }

    // Ensure it's really the identity in the mat-vec sense.
    // Apply the identity to psi via obvious mat-vec
    NSL::Tensor<ComplexType> Ipsi(nt, nx);
    for(int t=0; t < nt; t++){
        for(int x=0; x < nx; x++){
            for(int i=0; i< nt; i++){
                for(int y=0; y<nx; y++){
                    Ipsi(t,x) += identity(t,x,i,y) * psi(i,y);
                }
            }
        }
    }
    // If it's really the identity then nothing should have changed.
    REQUIRE( (Ipsi == psi).all() );

    // Then we apply Mdagger to each column.
    for(int i = 0; i < nt; i++){
        for(int y = 0; y < nx; y++){
            dense(NSL::Slice(), NSL::Slice(), i, y) = M.Mdagger(identity(NSL::Slice(), NSL::Slice(), i, y));
        }
    }
    
    // So, we can compare M†.I to (M.I)†
    REQUIRE( almost_equal(Mdense_dagger-dense, ComplexType(0,0)).all() );
    // This REQUIREment looks funny; why not just check that the two tensors are almost_equal directly, as in
    //      REQUIRE( almost_equal(Mdense_dagger, dense).all() );
    // TODO: almost_equal of +0.0000... and -0.0000... evaluates to False and that's extremely misleading
    // and this sign difference shows up when we construct these two tensors in different ways.



    // Finally, compare two ways of computing M†ψ
    NSL::Tensor<ComplexType>Mdense_dagger_psi(nt, nx), M_dagger_psi(nt, nx);
    // by doing the obvious mat-vec,
    for(int t=0; t < nt; t++){
        for(int x=0; x < nx; x++){
            for(int i=0; i< nt; i++){
                for(int y=0; y<nx; y++){
                    Mdense_dagger_psi(t,x) += Mdense_dagger(t,x,i,y) * psi(i,y);
                }
            }
        }
    }
    // and by applying Mdagger
    M_dagger_psi = M.Mdagger(psi);

    REQUIRE( almost_equal(Mdense_dagger_psi, M_dagger_psi).all() );
}

// ======================================================================
// Implementation Details: test_MdaggerM
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MdaggerM(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {

    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    //hardcoding the calculation done in the method MdaggerM of fermionMatrixHubbardExp class
    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(nt, nx);
    phi.rand();
    psi.rand();

    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);
    ComplexType I={0,1};
 
    auto direct = M.MdaggerM(psi);
    auto indirect = M.Mdagger(M.M(psi));

    INFO("nt: "+NSL::to_string(nt)+" nx: "+NSL::to_string(nx));

    REQUIRE( almost_equal(direct,indirect, std::numeric_limits<Type>::digits10-1).all() );
}

// ======================================================================
// Implementation Details: test_MdaggerM
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MMdagger(const NSL::size_t nt, LatticeType & Lattice, const Type & beta) {
    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(nt, nx);
    phi.rand();
    psi.rand();

    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);
    ComplexType I={0,1};
 
    auto direct = M.MMdagger(psi);
    auto indirect = M.M(M.Mdagger(psi));
    auto diff = direct - indirect;

    INFO("nt: "+NSL::to_string(nt)+" nx: "+NSL::to_string(nx));
    REQUIRE( almost_equal(direct,indirect, std::numeric_limits<Type>::digits10-1).all() );
}

// ======================================================================
// Implementation Details: test_M_batched
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_M_batched(const NSL::size_t nt, LatticeType & Lattice, const Type & beta){
    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    NSL::size_t Nbatch = 10;

    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(Nbatch, nt, nx);
    phi.rand();
    psi.rand();

    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);
    ComplexType I={0,1};
 
    auto direct = M.M(psi);

    for(NSL::size_t n = 0; n < Nbatch; n++){
        auto indirect = M.M(psi(n,NSL::Slice(),NSL::Slice()));
        REQUIRE( almost_equal(direct(n,NSL::Slice(),NSL::Slice()),indirect).all() );
    }
}

// ======================================================================
// Implementation Details: test_Mdagger_batched
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_Mdagger_batched(const NSL::size_t nt, LatticeType & Lattice, const Type & beta){
    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    NSL::size_t Nbatch = 10;

    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(Nbatch, nt, nx);
    phi.rand();
    psi.rand();

    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);
    ComplexType I={0,1};
 
    auto direct = M.Mdagger(psi);

    for(NSL::size_t n = 0; n < Nbatch; n++){
        auto indirect = M.Mdagger(psi(n,NSL::Slice(),NSL::Slice()));
        REQUIRE( almost_equal(direct(n,NSL::Slice(),NSL::Slice()),indirect).all() );
    }
}

// ======================================================================
// Implementation Details: test_MMdagger_batched
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MMdagger_batched(const NSL::size_t nt, LatticeType & Lattice, const Type & beta){
    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    NSL::size_t Nbatch = 10;

    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(Nbatch, nt, nx);
    phi.rand();
    psi.rand();

    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);
    ComplexType I={0,1};
 
    auto direct = M.MMdagger(psi);

    for(NSL::size_t n = 0; n < Nbatch; n++){
        auto indirect = M.MMdagger(psi(n,NSL::Slice(),NSL::Slice()));
        REQUIRE( almost_equal(direct(n,NSL::Slice(),NSL::Slice()),indirect).all() );
    }
}

// ======================================================================
// Implementation Details: test_MdaggerM_batched
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_fermionMatrixHubbardExp_MdaggerM_batched(const NSL::size_t nt, LatticeType & Lattice, const Type & beta){
    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();

    NSL::size_t Nbatch = 10;

    NSL::Tensor<Type> phi(nt, nx);
    NSL::Tensor<Type> psi(Nbatch, nt, nx);
    phi.rand();
    psi.rand();

    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);
    ComplexType I={0,1};
 
    auto direct = M.MdaggerM(psi);

    for(NSL::size_t n = 0; n < Nbatch; n++){
        auto indirect = M.MdaggerM(psi(n,NSL::Slice(),NSL::Slice()));
        REQUIRE( almost_equal(direct(n,NSL::Slice(),NSL::Slice()),indirect).all() );
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

    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);
    NSL::FermionMatrix::HubbardExp Mshift(Lattice,nt,beta);
    Mshift.populate(phiShift);

    Type result = M.logDetM();
    Type result_shift = Mshift.logDetM();

    INFO("nx: "+std::to_string(nx)+" nt: "+std::to_string(nt));
    INFO("result unshifted: "+NSL::to_string(result));
    INFO("result   shifted: "+NSL::to_string(result_shift));
    INFO("difference      : "+NSL::to_string(result-result_shift));

    REQUIRE(almost_equal(result_shift,result,std::numeric_limits<Type>::digits10-1));
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

    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);
    NSL::FermionMatrix::HubbardExp Mshift(Lattice,nt,beta);
    Mshift.populate(phiShift);

    Type result = M.logDetM();
    Type result_shift = Mshift.logDetM();

    INFO("nx: "+std::to_string(nx)+" nt: "+std::to_string(nt));
    INFO("result unshifted: "+NSL::to_string(result));
    INFO("result   shifted: "+NSL::to_string(result_shift));
    INFO("difference      : "+NSL::to_string(result-result_shift));

    REQUIRE(almost_equal(result_shift,result,std::numeric_limits<Type>::digits10-1));

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
    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);
    
    //When phi=0, logDetM = logdet(1 + exp_hopping_matrix(beta))
    Type result_exa = NSL::LinAlg::logdet(
            NSL::Matrix::Identity<Type>(nx) + Lattice.exp_hopping_matrix(beta)
    );

    Type result_alg = M.logDetM();

    INFO("nx: "+std::to_string(nx)+" nt: "+std::to_string(nt));
    INFO("result algorithm: "+NSL::to_string(result_alg));
    INFO("result exact    : "+NSL::to_string(result_exa));
    INFO("difference      : "+NSL::to_string(result_exa-result_alg));

    REQUIRE(almost_equal(result_alg,result_exa,std::numeric_limits<Type>::digits10-1));
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
    NSL::Tensor<Type> tmp (nt); tmp.rand();
    for(int i=0; i<nt; i++){
        phi(i, NSL::Slice() ) = tmp(i) ;
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
    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);
    Type result_alg = M.logDetM();
    Type result_exa = NSL::LinAlg::logdet(
            NSL::Matrix::Identity<Type>(nx) + 
            expsum * Lattice.exp_hopping_matrix(beta)
    );

    INFO("nx: "+std::to_string(nx)+" nt: "+std::to_string(nt));
    INFO("result algorithm: "+NSL::to_string(result_alg));
    INFO("result exact    : "+NSL::to_string(result_exa));
    INFO("difference      : "+NSL::to_string(result_exa-result_alg));

    // The double precision works fine with - 1 however, the floating point
    // requires a digit less precision hence the -2.
    // It is adviced to use float only if preciseness doesn't matter to much.
    REQUIRE(almost_equal(result_alg,result_exa, std::numeric_limits<Type>::digits10 - 2));

}


//Test cases

