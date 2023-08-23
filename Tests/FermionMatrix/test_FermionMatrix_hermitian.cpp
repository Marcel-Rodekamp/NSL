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

template<NSL::Concept::isComplex Type>
void test_FermionMatrix_U1Wilson_MMdagger_hermitian(NSL::size_t nt, NSL::size_t nx);

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


COMPLEX_NSL_TEST_CASE( "FermionMatrix U1 Wilson: MMdagger hermitian", "[FermionMatrix, MMdagger_hermitian]" ) {
    const NSL::size_t nt = GENERATE(2);//, 4, 6, 8, 10, 12, 14);
    const NSL::size_t nx = GENERATE(2);//, 4, 6, 8, 10, 12, 14);
    test_FermionMatrix_U1Wilson_MMdagger_hermitian<TestType>(nt, nx);
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

    INFO(fmt::format("MdaggerM.sum() = {}", NSL::to_string(MdaggerM.sum())));
    INFO(fmt::format("MdaggerMH.sum() = {}", NSL::to_string(MdaggerMH.sum())));
    INFO(
        NSL::LinAlg::max( 
            NSL::LinAlg::abs(MdaggerM - MdaggerMH)
        )
    );

	REQUIRE(almost_equal(MdaggerM, MdaggerMH, std::numeric_limits<Type>::digits10-2).all());
    REQUIRE(MdaggerMH.data() != MdaggerM.data());
}

//Test if MMdagger is hermitian
template<NSL::Concept::isComplex Type>
void test_FermionMatrix_U1Wilson_MMdagger_hermitian(NSL::size_t nt, NSL::size_t nx) {
    NSL::size_t dim = 2;
    NSL::complex<NSL::RealTypeOf<Type>> I{0,1};

    NSL::Lattice::Square<Type> lattice({nt,nx});
    NSL::Parameter params;
    params.addParameter<NSL::size_t>( "Nt", nt );
    params.addParameter<NSL::size_t>( "Nx", nx );
    params.addParameter<NSL::size_t>( "dim", dim );
    params.addParameter<Type>( "bare mass", 2 );
    params.addParameter<NSL::Device>( "device", NSL::CPU() );
    params.addParameter<NSL::Lattice::Square<Type>>("lattice",lattice);

	//NSL::Tensor<Type> U = NSL::LinAlg::exp( I*NSL::randn<Type>(nt,nx,dim) );
	NSL::Tensor<Type> U(nt,nx,dim); 
    U = NSL::LinAlg::exp(I*NSL::randn_like(U));
	//generate random fermion matrix for given lattice
    NSL::FermionMatrix::U1::Wilson<Type> M(params);
    M.populate(U);

	// Then we apply MMdagger to each column to generate the full matrix.
	NSL::Tensor<Type> psi(nt, nx, dim);
	NSL::Tensor<Type> MMdagger(nt, nx, dim, nt, nx, dim);

	for(int i = 0; i < nt; i++){
		for(int y = 0; y < nx; y++){
            for (NSL::size_t d = 0; d < dim; ++d){
                psi(i,y,d) = 1;
			    MMdagger(NSL::Slice(), NSL::Slice(), NSL::Slice(),i, y, d) = M.MMdagger(
                    psi
                );
                psi(i,y,d) = 0;
            }
		}
	}

	// Finally we construct the adjoint matrix and ensure it is equal to the original
	NSL::Tensor<Type> MMdaggerH = MMdagger.T(0,3).T(1,4).T(2,5).conj();


    std::cout << MMdagger.real() << std::endl;
    std::cout << MMdaggerH.real() << std::endl;
    REQUIRE(MMdaggerH.data() != MMdagger.data());
	//REQUIRE(almost_equal(MMdagger, MMdaggerH, std::numeric_limits<Type>::digits10).all());
    for(int i = 0; i < nt; i++){
	    for(int y = 0; y < nx; y++){
            for (NSL::size_t d = 0; d < dim; ++d){
                for(int j = 0; j < nt; j++){
                	for(int z = 0; z < nx; z++){
                        for (NSL::size_t b = 0; b < dim; ++b){
                            INFO(fmt::format("i={},y={},d={},j={},z={},b={}",i,y,d,j,z,b));
                            INFO(fmt::format("MMdagger(i,y,d,j,z,b) ={: 16f}+i{: 16f}",NSL::real(MMdagger(i,y,d,j,z,b)),NSL::imag(MMdagger(i,y,d,j,z,b))));
                            INFO(fmt::format("MMdaggerH(i,y,d,j,z,b)={: 16f}+i{: 16f}",NSL::real(MMdaggerH(i,y,d,j,z,b)),NSL::imag(MMdaggerH(i,y,d,j,z,b))));
                            REQUIRE( MMdaggerH(i,y,d,j,z,b) == NSL::LinAlg::conj(MMdagger(j,z,b,i,y,d) ));     
                            REQUIRE( almost_equal(MMdagger(i,y,d,j,z,b), MMdaggerH(i,y,d,j,z,b)) );    
                        }
                    }
                }
            }
        }
    }
}

//Test cases

