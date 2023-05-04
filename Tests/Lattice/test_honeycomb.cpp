#include "../test.hpp"

/*! \file test_honeycomb.cpp
 *  Test the `Lattice::Honeycomb` implementation.
 */

//! Torch requirement
using size_type = int;

/*!
 *  \param size the number of sites
 *  \param kappa the counter-clockwise hopping amplitude
 **/
template<typename T>
void test_honeycomb_volume(NSL::Lattice::Honeycomb<T> & lattice){
    // 2 sites per unit cell
    REQUIRE(lattice.sites() == 2*lattice.unit_cells);
}

template<typename T>
void test_honeycomb_valence(NSL::Lattice::Honeycomb<T> & lattice){
    // each vertex is valence 3
    INFO( lattice.adjacency_matrix().sum(0) );
    REQUIRE( (lattice.adjacency_matrix().sum(0) == 3).all() );
    INFO( lattice.adjacency_matrix().sum(1) );
    REQUIRE( (lattice.adjacency_matrix().sum(1) == 3).all() );
}

template<typename T>
void test_honeycomb_hermiticity(NSL::Lattice::Honeycomb<T> & lattice){
    // The hopping matrix is Hermitian
    NSL::Tensor<T> K = lattice.hopping_matrix();
    REQUIRE( ((K - K.conj().transpose(0,1)) == 0).all() );
}

template<typename T>
void test_honeycomb_eigenvector_orthogonality(const NSL::Tensor<T> & eigenvectors){
    
    NSL::size_t n = eigenvectors.shape(0);
    NSL::Tensor<T> one = NSL::Matrix::Identity<T>(n);

    NSL::Tensor<T> dot_products = NSL::LinAlg::mat_mul(NSL::LinAlg::transpose(eigenvectors).conj(), eigenvectors);
    REQUIRE( almost_equal(one, dot_products).all() );
}

template<typename T>
void test_honeycomb_spectrum_limits(const NSL::Tensor<T> & eigenvalues, const double epsilon=1.e-6){

    double limit = 3 + epsilon;

    INFO("Limit = " << limit);
    
    auto not_too_small = (-limit <= eigenvalues);
    auto not_too_big   = (eigenvalues <= +limit);

    INFO("Eigenvalues <= +3" << not_too_big);
    REQUIRE( not_too_big.all() );

    INFO("-3 <= Eigenvalues" << not_too_small);
    REQUIRE( not_too_small.all() );

}

template<typename T>
void test_honeycomb_dirac_points(const size_type & L1, const size_type & L2, const NSL::Tensor<T> & eigenvalues){

    // We have to cast to into to get a correct sum.
    int zero_energy_states = NSL::Tensor<int>(almost_equal(eigenvalues, T(0))).sum();

    // If (L1, L2) == (0,0) mod 3 we get 2 Dirac points.
    // Each Dirac point has two energy eigenvectors; one of each band.
    // So if there are Dirac points we expect to see 4 zero-energy states; and 0 otherwise.
    if( (L1 % 3 == 0) && (L2 % 3 == 0) ){
        REQUIRE( zero_energy_states == 4 );
    } else {
        REQUIRE( zero_energy_states == 0 );
    }

}

// =============================================================================
// Test Cases
// =============================================================================

REAL_NSL_TEST_CASE( "Lattice: Honeycomb", "[Lattice, Honeycomb]" ) {
    const size_type L1 = GENERATE(2, 3, 4, 5, 6, 9, 12);
    const size_type L2 = GENERATE(2, 3, 4, 5, 6, 9, 12);
    //const size_type L2 = L1;

    std::vector<size_type> L(2);
    L[0] = L1;
    L[1] = L2;

    INFO("Type = " << typeid(TestType).name());
    INFO("L    = " << L);

    NSL::Lattice::Honeycomb<TestType> lattice(L);
    auto [energies, vectors]  = NSL::LinAlg::eigh(lattice.hopping_matrix(1.0));
    
    INFO(lattice.name());
    INFO("Sites = " << lattice.sites());

    INFO("Coordinates")
	INFO("-----------");
	INFO(lattice.coordinates());
	INFO("-----------");
	
    // INFO("Adjacency matrix:\n" << lattice.adjacency_matrix());
	
    INFO("Energy eigenvalues");
    INFO("------------------");
    INFO(energies);
    INFO("------------------");
    
    test_honeycomb_volume<TestType>(lattice);
    test_honeycomb_valence<TestType>(lattice);
    test_honeycomb_hermiticity<TestType>(lattice);
    test_honeycomb_eigenvector_orthogonality(vectors);
    test_honeycomb_spectrum_limits(energies);
    test_honeycomb_dirac_points(L1, L2, energies);
}

