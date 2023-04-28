#include "../test.hpp"

/*! \file test_square.cpp
 *  Test the `Lattice::Square` implementation.
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

// =============================================================================
// Test Cases
// =============================================================================

REAL_NSL_TEST_CASE( "Lattice: Honeycomb", "[Lattice, Honeycomb]" ) {
    const size_type L1 = GENERATE(2, 3, 4, 6, 12, 15);
    //const size_type L2 = GENERATE(2, 3, 4, 6, 12, 15);
    const size_type L2 = L1;

    std::vector<size_type> L(2);
    L[0] = L1;
    L[1] = L2;

    INFO("Type = " << typeid(TestType).name());
    INFO("L    = " << L);

    NSL::Lattice::Honeycomb<TestType> lattice(L);

    INFO(lattice.name());
    INFO(lattice.sites());
    INFO(lattice.coordinates());
    INFO(lattice.adjacency_matrix());
    
    test_honeycomb_volume<TestType>(lattice);
    test_honeycomb_valence<TestType>(lattice);
    test_honeycomb_hermiticity<TestType>(lattice);
    //! \todo: Add a test of the spectrum; Dirac points should only show up when L%3 = 0.
}

