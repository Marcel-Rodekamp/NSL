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
    INFO(lattice.sites());

    size_t volume=1;
    for(const auto& value: size) volume*=value;
    REQUIRE(lattice.sites() == volume);

    //INFO(lattice.coordinates());
    //INFO(lattice.adjacency_matrix());

    //! \todo Require that FT * hopping * FT† is diagonal and sensible.

    REQUIRE( false );
}

/*!
 *  \param size the number of sites
 *  \param kappa the counter-clockwise hopping amplitude
 **/
template<typename T>
void test_cube(const size_type & size, T kappa = 1.){
    INFO("Type = " << typeid(T).name());
    INFO("size = " << size);

    NSL::Lattice::Cube3D<T> lattice(size);
    INFO(lattice.name());
    INFO(lattice.sites());

    size_t volume=size * size * size;
    REQUIRE(lattice.sites() == volume);

    //INFO(lattice.coordinates());
    //INFO(lattice.adjacency_matrix());

    //! \todo Require that FT * hopping * FT† is diagonal and sensible.

    REQUIRE( false );
}

// =============================================================================
// Test Cases
// =============================================================================

TEST_CASE( "Lattice: 1D Square", "[Lattice, Square, 1D]" ) {
    const size_type d0 = GENERATE(2, 4, 8, 16);

    std::vector<size_type> n(1);
    n[0] = d0;
    // floating point types
    test_square<float>(n);
    //! \todo add tests of complex amplitudes
    // NOTE: no complex<type>s 
    // because the hopping amplitude
    // wouldn't be hermitian, which is required.
    // A (generic) FIX would require a complex conjugation 
    // on real float, double for Square::hops_.
}

TEST_CASE( "Lattice: 2D Square", "[Lattice, Square, 2D]" ) {
    const size_type d0 = GENERATE(2, 3, 4, 8, 16);
    const size_type d1 = GENERATE(2, 4, 5, 8, 16);

    std::vector<size_type> n(2);
    n[0] = d0;
    n[1] = d1;
    // floating point types
    test_square<float>(n);
    test_square<double>(n);
    //! \todo add tests of complex amplitudes
    // NOTE: no complex<type>s 
    // because the hopping amplitude
    // wouldn't be hermitian, which is required.
    // A (generic) FIX would require a complex conjugation 
    // on real float, double for Square::hops_.
}

TEST_CASE( "Lattice: 3D Square", "[Lattice, Square, 3D]" ) {
    const size_type d0 = GENERATE(2, 3, 4, 8, 16);
    const size_type d1 = GENERATE(2, 4, 5, 8, 16);
    const size_type d2 = GENERATE(2, 4, 8, 9, 16);

    std::vector<size_type> n(3);
    n[0] = d0;
    n[1] = d1;
    n[2] = d2;
    // floating point types
    test_square<float>(n);
    test_square<double>(n);
    //! \todo add tests of complex amplitudes
    // NOTE: no complex<type>s 
    // because the hopping amplitude
    // wouldn't be hermitian, which is required.
    // A (generic) FIX would require a complex conjugation 
    // on real float, double for Square::hops_.
}

TEST_CASE( "Lattice: Cube", "[Lattice, Cube, 3D]" ) {
    const size_type n = GENERATE(2, 3, 4, 8, 16);

    // floating point types
    test_cube<float>(n);
    test_cube<double>(n);
    //! \todo add tests of complex amplitudes
    // NOTE: no complex<type>s 
    // because the hopping amplitude
    // wouldn't be hermitian, which is required.
    // A (generic) FIX would require a complex conjugation 
    // on real float, double for Square::hops_.
}
