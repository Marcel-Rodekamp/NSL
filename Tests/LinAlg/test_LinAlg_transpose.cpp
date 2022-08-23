#include "../test.hpp"

// Torch requirement
using size_type = long int;

template<typename T>
void test_transpose(const size_type & size){
    NSL::Tensor<T> t(size, size);
    t.rand();

    NSL::Tensor<T> transposed = NSL::LinAlg::transpose(t);

    REQUIRE( (transposed == t.transpose()).all() );

}

template<typename T>
void test_stacked_transpose(const size_type & size){
    NSL::Tensor<T> t(10, size, size);
    t.rand();

    NSL::Tensor<T> transposed = NSL::LinAlg::transpose(t);

    REQUIRE( (transposed == t.transpose()).all() );

}

template<typename T>
void test_stacked_transpose(const size_type & size, size_type dim0, size_type dim1){
    NSL::Tensor<T> t(10, size, size);
    t.rand();

    NSL::Tensor<T> transposed = NSL::LinAlg::transpose(t, dim0, dim1);

    REQUIRE( (transposed == t.transpose(dim0, dim1)).all() );

}

// =============================================================================
// Test Cases
// =============================================================================

FLOAT_NSL_TEST_CASE( "LinAlg: Transpose", "[LinAlg,transpose,default]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);
    test_transpose<TestType>(size);
}


FLOAT_NSL_TEST_CASE( "LinAlg: Stacked Transpose", "[LinAlg,transpose,stacked,default]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);
    test_stacked_transpose<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "LinAlg: Stacked Transpose of axes that aren't last", "[LinAlg,transpose,stacked,dims]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);
    test_stacked_transpose<TestType>(size,0,1);
    test_stacked_transpose<TestType>(size,0,2);
}
