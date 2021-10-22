//#include <complex>
#include "complex.hpp"
#include "../test.hpp"
#include <typeinfo>
#include "LinAlg/abs.hpp"
#include "Lattice/Implementations/complete.hpp"


/*! \file test_complete.cpp
 *  Test the `Lattice::Complete` implementation.
 */

//! Torch requirement
using size_type = std::size_t;

/*!
 *  \param size the number of sites
 **/
template<typename T>
void test_complete(NSL::Lattice::SpatialLattice<T> & lattice){
    INFO("Type = " << typeid(T).name());
    INFO(lattice.name());

    REQUIRE(lattice.bipartite() == (lattice.sites() <= 2));

    auto coordinates = lattice.coordinates();
    NSL::Tensor<double> centroid = coordinates.sum(0);  // sum all the respective components
    INFO(centroid);
    REQUIRE((centroid < 1.e-12).all());

    // Every site is connected to every site that isn't itself:
    auto adj = lattice.adjacency_matrix();
    INFO(adj);
    REQUIRE( (adj.sum(0) == lattice.sites() - 1).all());
    REQUIRE( (adj.sum(1) == lattice.sites() - 1).all());
}

// =============================================================================
// Test Cases
// =============================================================================

REAL_NSL_TEST_CASE( "Lattice: Complete", "[Lattice, Complete]" ) {
    const size_type size = GENERATE(2, 3, 4, 5, 8, 101, 202, 505, 1010);
    NSL::Lattice::Complete<TestType> lattice(size);
    test_complete<TestType>(lattice);
}

REAL_NSL_TEST_CASE( "Lattice: Triangle", "[Lattice, Triangle]" ) {
    NSL::Lattice::Triangle<TestType> lattice;

    REQUIRE(lattice.sites() == 3);
    test_complete<TestType>(lattice);
}

REAL_NSL_TEST_CASE( "Lattice: Tetrahedron", "[Lattice, Tetrahedron]" ) {
    NSL::Lattice::Tetrahedron<TestType> lattice;

    REQUIRE(lattice.sites() == 4);
    test_complete<TestType>(lattice);
}
