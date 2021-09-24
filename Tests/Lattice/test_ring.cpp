//#include <complex>
#include "complex.hpp"
#include "catch2/catch.hpp"
#include <typeinfo>
#include "Lattice/Implementations/ring.hpp"

// Torch requirement
using size_type = std::size_t;

template<typename T>
void test_ring(const size_type & size){
    INFO("Type = " << typeid(T).name());
    INFO("size = " << size);

    NSL::Lattice::Ring<T> ring(size);
    INFO(ring.name());

    REQUIRE(ring.sites() == size);

    //! \bug yields `SIGABRT - Abort (abnormal termination) signal`
    // I think it's that the Site.coordinates = calculated coordinates
    // in the Lattice::Ring constructor isn't handled correctly.
    INFO(ring(0).coordinates)

    // Require that adjacency^size has a diagonal of 2.
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
    const size_type size = GENERATE(1, 100, 200, 500, 1000);

    // floating point types
    test_ring<float>(size);
    test_ring<double>(size);
    // NOTE: no complex<type>s 
    // because the hopping amplitude
    // wouldn't be hermitian, which is required.
    // A (generic) FIX would require a complex conjugation 
    // on real float, double for Ring::hops_.
}
