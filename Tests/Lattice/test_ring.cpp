//#include <complex>
#include "complex.hpp"
#include "catch2/catch.hpp"
#include <typeinfo>
#include "Lattice/Implementations/ring.hpp"


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

    INFO(ring.coordinates());
    //! \todo Sum up the coordinates; they should evenly surround the origin.

    //  Perhaps make these their own stand-alone tests.
    //! \todo Require that adjacency^size has a diagonal of 2.
    //! \todo Require that FT * hopping * FT† is diagonal and sensible.
}

// =============================================================================
// Test Cases
// =============================================================================

// short int                Not Supported by torch
//unsigned short int        Not Supported by torch
//unsigned int              Not Supported by torch
//size_type                  Not Supported by torch
//unsigned size_type         Not Supported by torch
//long size_type             Not Supported by torch
//unsigned long size_type    Not Supported by torch
//long double               Not Supported by torch
//NSL::complex<int>         Not Supported by torch

TEST_CASE( "Lattice: Ring", "[Lattice, Ring]" ) {
    const size_type size = GENERATE(2, 4, 8, 101, 202, 505, 1010);

    // floating point types
    test_ring<float>(size, 0.5);
    test_ring<double>(size, 2.0);
    //! \todo add tests of complex amplitudes
    // NOTE: no complex<type>s 
    // because the hopping amplitude
    // wouldn't be hermitian, which is required.
    // A (generic) FIX would require a complex conjugation 
    // on real float, double for Ring::hops_.
}
