#include "../test.hpp"
#include <iomanip>
#include <limits>

// We rely on the accuracy of the application of M for testing.
// If M is implemented correctly and M_dense is implemented correctly we can
// use the dense implementation to check that Mdagger is implemented correctly.
// THEN we can use the correctness of M and Mdagger to check if MMdagger and MdaggerM are correct,
// since in principle the result of MMdagger should just be M•Mdagger (and similarly for MdaggerM).

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_FermionMatrix_MMdagger_hermitian(const NSL::size_t nt, LatticeType & Lattice, const std::string latticeName, const Type & beta = 2);

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_FermionMatrix_MdaggerM_hermitian(const NSL::size_t nt, LatticeType & Lattice, const std::string latticeName, const Type & beta = 2);


COMPLEX_NSL_TEST_CASE( "FermionMatrix: MMdagger hermitian", "[FermionMatrix, MMdagger_hermitian]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 6, 8, 10, 12, 14);
    const NSL::size_t nx = GENERATE(2, 4, 6, 8, 10, 12, 14);

    NSL::Lattice::Ring<TestType> LatticeR(nx);
    test_FermionMatrix_MMdagger_hermitian<TestType>(nt, LatticeR, "Ring");

    NSL::Lattice::Square<TestType> LatticeS({nx, nx});
    test_FermionMatrix_MMdagger_hermitian<TestType>(nt, LatticeS, "Square");

    NSL::Lattice::Complete<TestType> LatticeC(nx);
    test_FermionMatrix_MMdagger_hermitian<TestType>(nt, LatticeC, "Complete");
    
}

COMPLEX_NSL_TEST_CASE( "FermionMatrix: MdaggerM hermitian", "[FermionMatrix, MdaggerM_hermitian]" ) {
    const NSL::size_t nt = GENERATE(2, 4, 6, 8, 10, 12, 14);
    const NSL::size_t nx = GENERATE(2, 4, 6, 8, 10, 12, 14);

    NSL::Lattice::Ring<TestType> LatticeR(nx);
    test_FermionMatrix_MdaggerM_hermitian<TestType>(nt, LatticeR, "Ring");

    NSL::Lattice::Square<TestType> LatticeS({nx, nx});
    test_FermionMatrix_MdaggerM_hermitian<TestType>(nt, LatticeS, "Square");

    NSL::Lattice::Complete<TestType> LatticeC(nx);
    test_FermionMatrix_MdaggerM_hermitian<TestType>(nt, LatticeC, "Complete");
}

// ======================================================================
// Implementation Details: FermionMatrix_MMdagger_hermitian
// ======================================================================

//Test if MMdagger is hermitian
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_FermionMatrix_MMdagger_hermitian(const NSL::size_t nt, LatticeType & Lattice, const std::string latticeName, const Type & beta) {
    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();
	NSL::Tensor<Type> phi(nt, nx);
    phi.rand();
	//generate random fermion matrix for given lattice
    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);

    INFO("lattice: "+latticeName+" nt: "+NSL::to_string(nt)+" nx: "+NSL::to_string(nx));

	// First we construct the identity matrix.
	NSL::Tensor<Type> identity(nt, nx, nt, nx);
	for(int t = 0; t < nt; t++){
		identity(t,NSL::Slice(), t, NSL::Slice()) = NSL::Matrix::Identity<Type>(nx);
	}

	// Then we apply MMdagger to each column to generate the full matrix.
	NSL::Tensor<Type> MMdagger(nt, nx, nt, nx);
	for(int i = 0; i < nt; i++){
		for(int y = 0; y < nx; y++){
			MMdagger(NSL::Slice(), NSL::Slice(), i, y) = M.MMdagger(identity(NSL::Slice(), NSL::Slice(), i, y));
		}
	}

	// Finally we construct the adjoint matrix and ensure it is equal to the original
	NSL::Tensor<Type> MMdaggerH = MMdagger.T(0,2).H(1,3);

    REQUIRE(MMdaggerH.data() != MMdagger.data());
	REQUIRE(almost_equal(MMdagger, MMdaggerH, std::numeric_limits<Type>::digits10).all());
}


// ======================================================================
// Implementation Details: FermionMatrix_MdaggerM_hermitian
// ======================================================================

//Test if MdaggerM is hermitian
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void test_FermionMatrix_MdaggerM_hermitian(const NSL::size_t nt, LatticeType & Lattice, const std::string latticeName, const Type & beta) {
    typedef NSL::complex<typename NSL::RT_extractor<Type>::value_type> ComplexType;
    NSL::size_t nx = Lattice.sites();
	NSL::Tensor<Type> phi(nt, nx);
    phi.rand();
	//generate random fermion matrix for given lattice
    NSL::FermionMatrix::HubbardExp M(Lattice,nt,beta);
    M.populate(phi);
    
    INFO("lattice: "+latticeName+" nt: "+NSL::to_string(nt)+" nx: "+NSL::to_string(nx));

	// First we construct the identity matrix.
	NSL::Tensor<Type> identity(nt, nx, nt, nx);
	for(int t = 0; t < nt; t++){
		identity(t,NSL::Slice(), t, NSL::Slice()) = NSL::Matrix::Identity<Type>(nx);
	}

	// Then we apply MdaggerM to each column to generate the full matrix.
	NSL::Tensor<Type> MdaggerM(nt, nx, nt, nx);
	for(int i = 0; i < nt; i++){
		for(int y = 0; y < nx; y++){
			MdaggerM(NSL::Slice(), NSL::Slice(), i, y) = M.MdaggerM(identity(NSL::Slice(), NSL::Slice(), i, y));
		}
	}

	// Finally we construct the adjoint matrix and ensure it is equal to the original
	NSL::Tensor<Type> MdaggerMH = MdaggerM.T(0,2).H(1,3);

    REQUIRE(MdaggerMH.data() != MdaggerM.data());
	REQUIRE(almost_equal(MdaggerM, MdaggerMH, std::numeric_limits<Type>::digits10).all());
}

//Test cases

