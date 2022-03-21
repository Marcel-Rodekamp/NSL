#include "../test.hpp"

/*! \file test_ring.cpp
 *  Test the `Lattice::Ring` implementation.
 */

//! Torch requirement
using size_type = std::size_t;

/*! Test bipartiteness algorithm on rings, for which the answer is easy.
 *  \param size the number of sites
 **/
template<typename T>
void test_ring(const size_type & size){
    INFO("Type = " << typeid(T).name());

    NSL::Lattice::Ring<T> lattice(size);
    INFO(lattice.name());
    INFO("size = " << lattice.sites());

    REQUIRE(lattice.bipartite() == (size%2 == 0));
}

/*! Test bipartiteness algorithm on 2D square lattices, for which the answer is easy.
 *  \param dim0 the number of sites in the 0th direction
 *  \param dim1 the number of sites in the 1st direction
 **/
template<typename T>
void test_square2D(const size_type & dim0, const size_type & dim1){
    INFO("Type = " << typeid(T).name());

    NSL::Lattice::Square<T> lattice({dim0, dim1});
    INFO(lattice.name());
    INFO("size = " << lattice.sites());

    // \todo swap these tests when periodic boundary conditions are implemented:
    // \todo This test is only meaningful if bipartite() is not overridden by the Square implementation---which it probably SHOULD be eventually.
    // REQUIRE(lattice.bipartite() == ((dim0%2 == 0) && (dim1%2 == 0)));
    REQUIRE(lattice.bipartite());
}

/*! Test bipartiteness algorithm on 3D square lattices, for which the answer is easy.
 *  \param dim0 the number of sites in the 0th direction
 *  \param dim1 the number of sites in the 1st direction
 *  \param dim1 the number of sites in the 2nd direction
 **/
template<typename T>
void test_square3D(const size_type & dim0, const size_type & dim1, const size_type & dim2){
    INFO("Type = " << typeid(T).name());

    NSL::Lattice::Square<T> lattice({dim0, dim1, dim2});
    INFO(lattice.name());
    INFO("size = " << lattice.sites());

    // \todo swap these tests when periodic boundary conditions are implemented:
    // \todo This test is only meaningful if bipartite() is not overridden by the Square implementation---which it probably SHOULD be eventually.
    // REQUIRE(lattice.bipartite() == ((dim0%2 == 0) && (dim1%2 == 0) && (dim2%2 == 0));
    REQUIRE(lattice.bipartite());
}

// =============================================================================
// Test Cases
// =============================================================================

REAL_NSL_TEST_CASE( "Bipartite: Ring", "[Bipartite, Ring]" ) {
    const size_type size = GENERATE(2, 3, 4, 5, 100, 101, 202, 505);
    test_ring<TestType>(size);
}

REAL_NSL_TEST_CASE( "Bipartite: Square 2D", "[Bipartite, Square, 2D]" ) {
    const size_type d0 = GENERATE(2, 3, 4, 5);
    const size_type d1 = GENERATE(2, 3, 4, 5);
    test_square2D<TestType>(d0, d1);
}

REAL_NSL_TEST_CASE( "Bipartite: Square 3D", "[Bipartite, Square, 3D]" ) {
    const size_type d0 = GENERATE(2, 3, 4, 5);
    const size_type d1 = GENERATE(2, 3, 4, 5);
    const size_type d2 = GENERATE(2, 3, 4, 5);
    test_square3D<TestType>(d0, d1, d2);
}
