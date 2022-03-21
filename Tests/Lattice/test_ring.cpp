#include "../test.hpp"

/*! \file test_ring.cpp
 *  Test the `Lattice::Ring` implementation.
 */

//! Torch requirement
using size_type = std::size_t;

/*!
 *  \param size the number of sites
 *  \param kappa the counter-clockwise hopping amplitude
 **/
template<typename T>
void test_ring(const size_type & size, T kappa = 1.){
    INFO("Type = " << typeid(T).name());
    INFO("size = " << size);

    NSL::Lattice::Ring<T> ring(size, kappa);
    INFO(ring.name());

    REQUIRE(ring.sites() == size);
    REQUIRE(ring.bipartite() == (size%2 == 0));

    NSL::Tensor<double> x = ring.coordinates();
    NSL::Tensor<double> COM = x.sum(0);
    double epsilon = 1e-12;
    REQUIRE( (COM.abs() <= epsilon).all() );

    //  Perhaps make these their own stand-alone tests.
    //! \todo Require that adjacency^size has a diagonal of 2.
    //! \todo Require that FT * hopping * FT† is diagonal and sensible.
}

// =============================================================================
// Test Cases
// =============================================================================

REAL_NSL_TEST_CASE( "Lattice: Ring", "[Lattice, Ring]" ) {
    const size_type size = GENERATE(2, 4, 8, 101, 202, 505, 1010);
    const TestType kappa = GENERATE(0.5, 2.0);

    test_ring<TestType>(size, kappa);
    test_ring<NSL::complex<TestType>>(size, kappa);
    test_ring<NSL::complex<TestType>>(size, NSL::complex<TestType>(kappa,kappa));
}
