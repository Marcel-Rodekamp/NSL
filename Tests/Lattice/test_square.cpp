//#include <complex>
//#include "complex.hpp"
#include "../test.hpp"
#include <typeinfo>
#include "Lattice/Implementations/square.hpp"


/*! \file test_square.cpp
 *  Test the `Lattice::Square` implementation.
 */

//! Torch requirement
using size_type = std::size_t;

/*!
 *  \param size the number of sites
 *  \param kappa the counter-clockwise hopping amplitude
 **/
template<typename T>
void test_square(const std::vector<size_type> & size, T kappa = 1.){
    INFO("Type = " << typeid(T).name());
    INFO("size = " << size);

    NSL::Lattice::Square<T> lattice(size);
    INFO(lattice.name());

    // REQUIRE(lattice.sites() == size);

    INFO(lattice.coordinates());
    //! \todo Sum up the coordinates; they should evenly surround the origin.

    //  Perhaps make these their own stand-alone tests.
    //! \todo Require that adjacency^size has a diagonal of 2.
    //! \todo Require that FT * hopping * FT† is diagonal and sensible.

    REQUIRE( false );
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

/*
TEST_CASE( "Lattice: Square", "[Lattice, Square, 2D]" ) {
    const size_type d0 = GENERATE(4);
    const size_type d1 = GENERATE(4);

    std::vector<size_type> n(2);
    n[0] = d0;
    n[1] = d1;
    // floating point types
    // test_ring<float>(size);
    test_square<double>(n);
    //! \todo add tests of complex amplitudes
    // NOTE: no complex<type>s 
    // because the hopping amplitude
    // wouldn't be hermitian, which is required.
    // A (generic) FIX would require a complex conjugation 
    // on real float, double for Ring::hops_.
}
*/

TEST_CASE( "Lattice: Square", "[Lattice, Square, 3D]" ) {
    const size_type d0 = GENERATE(4);
    const size_type d1 = GENERATE(4);
    const size_type d2 = GENERATE(2);

    std::vector<size_type> n(3);
    n[0] = d0;
    n[1] = d1;
    n[2] = d2;
    // floating point types
    // test_ring<float>(size);
    test_square<double>(n);
    //! \todo add tests of complex amplitudes
    // NOTE: no complex<type>s 
    // because the hopping amplitude
    // wouldn't be hermitian, which is required.
    // A (generic) FIX would require a complex conjugation 
    // on real float, double for Ring::hops_.
}
