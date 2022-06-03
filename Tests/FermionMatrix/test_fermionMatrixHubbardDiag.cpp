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





//Test cases