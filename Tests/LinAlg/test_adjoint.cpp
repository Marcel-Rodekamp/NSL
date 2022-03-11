#include "complex.hpp"
#include "../test.hpp"
#include "LinAlg/adjoint.hpp"
#include <typeinfo>

// Torch requirement
using size_type = long int;

template<typename T>
void test_adjoint(const size_type & size){
    NSL::Tensor<T> t(size, size);
    t.rand();

    NSL::Tensor<T> adjointed = NSL::LinAlg::adjoint(t);

    REQUIRE( (adjointed == t.adjoint()).all() );

}

template<typename T>
void test_stacked_adjoint(const size_type & size){
    NSL::Tensor<T> t(10, size, size);
    t.rand();

    NSL::Tensor<T> adjointed = NSL::LinAlg::adjoint(t);

    REQUIRE( (adjointed == t.adjoint()).all() );

}

template<typename T>
void test_stacked_adjoint(const size_type & size, size_type dim0, size_type dim1){
    NSL::Tensor<T> t(10, size, size);
    t.rand();

    NSL::Tensor<T> adjointed = NSL::LinAlg::adjoint(t, dim0, dim1);

    REQUIRE( (adjointed == t.adjoint(dim0, dim1)).all() );

}

// =============================================================================
// Test Cases
// =============================================================================

FLOAT_NSL_TEST_CASE( "LinAlg: adjoint", "[LinAlg,adjoint,default]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);
    test_adjoint<TestType>(size);
}


FLOAT_NSL_TEST_CASE( "LinAlg: Stacked adjoint", "[LinAlg,adjoint,stacked,default]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);
    test_stacked_adjoint<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "LinAlg: Stacked adjoint of axes that aren't last", "[LinAlg,adjoint,stacked,dims]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);
    test_stacked_adjoint<TestType>(size,0,1);
    test_stacked_adjoint<TestType>(size,0,2);
}

/*!
 * note: should really test for ints, bools too, but those cannot be filled with rand();
 *       It's not something necessarily 'trivial' to worry about because these things take up different amounts of memory.
 * */
