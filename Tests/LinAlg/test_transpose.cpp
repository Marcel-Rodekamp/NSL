#include "../test.hpp"

// Torch requirement
using size_type = long int;

// ======================================================================
// Implementation details: test_transpose
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_transpose(SizeTypes ... sizes){
    NSL::Tensor<Type> t(sizes...);
    t.rand();

    NSL::Tensor<Type> transposed = NSL::LinAlg::transpose(t);
    //Comparing with Tensor transpose
    REQUIRE( (transposed == t.transpose()).all() );

}

// ======================================================================
// Implementation details: test_transpose_memory
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_transpose_memory(SizeTypes ... sizes){
    NSL::Tensor<Type> t(sizes...);
    t.rand();

    NSL::Tensor<Type> transposed = NSL::LinAlg::transpose(t);
    //check that the memory spaces are disjunct    
    REQUIRE( (t.data() != transposed.data()).all() );

}
// ======================================================================
// Implementation details: test_stacked_transpose
// ======================================================================

template<typename T>
void test_stacked_transpose(const size_type & size){
    NSL::Tensor<T> t(10, size, size);
    t.rand();

    NSL::Tensor<T> transposed = NSL::LinAlg::transpose(t);

    REQUIRE( (transposed == t.transpose()).all() );

}

// ======================================================================
// Implementation details: test_stacked_transpose
// ======================================================================

template<typename T>
void test_stacked_transpose(const size_type & size, size_type dim0, size_type dim1){
    NSL::Tensor<T> t(10, size, size);
    t.rand();

    NSL::Tensor<T> transposed = NSL::LinAlg::transpose(t, dim0, dim1);

    REQUIRE( (transposed == t.transpose(dim0, dim1)).all() );

}

// ======================================================================
// Implementation details: test_transpose_of_transpose
// ======================================================================

template<typename T>
void test_transpose_of_transpose(const size_type & size){
    NSL::Tensor<T> t(10, size, size);
    t.rand();

    NSL::Tensor<T> doubletranspose = NSL::LinAlg::transpose(NSL::LinAlg::transpose(t));
    //Checking (A^T)^T = A
    REQUIRE( (doubletranspose == t).all() );

}

// ======================================================================
// Implementation details: test_transpose_matmul
// ======================================================================

template<typename T>
void test_transpose_matmul(const size_type & size){
    NSL::Tensor<T> a(size,size),b(size,size), abtrans, btransatrans;
    a.rand();
    b.rand();
    abtrans = NSL::LinAlg::transpose(NSL::LinAlg::mat_mul(a,b));
    btransatrans = NSL::LinAlg::mat_mul(NSL::LinAlg::transpose(b),NSL::LinAlg::transpose(a));
    //Checking (AB)^T = B^T A^T
    //Very low precision for double and float
    REQUIRE( almost_equal(abtrans, btransatrans, std::numeric_limits<T>::digits10-9).all() );

}
// =============================================================================
// Test Cases
// =============================================================================

FLOAT_NSL_TEST_CASE( "LinAlg: 1D Transpose", "[LinAlg,1D,transpose,default]" ) {
    const size_type size = GENERATE(2, 4, 8, 12, 16);
    test_transpose<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "LinAlg: 2D Transpose", "[LinAlg,2D,transpose,default]" ) {
    const size_type size_1 = GENERATE(2, 4, 8, 12, 16);
    const size_type size_2 = GENERATE(2, 4, 8, 12, 16);
    test_transpose<TestType>(size_1,size_2);
}

FLOAT_NSL_TEST_CASE( "LinAlg: 3D Transpose", "[LinAlg,3D,transpose,default]" ) {
    const size_type size_1 = GENERATE(2, 4, 8, 12, 16);
    const size_type size_2 = GENERATE(2, 4, 8, 12, 16);
    const size_type size_3 = GENERATE(2, 4, 8, 12, 16);
    test_transpose<TestType>(size_1,size_2,size_3);
}

FLOAT_NSL_TEST_CASE( "LinAlg: Stacked Transpose", "[LinAlg,transpose,stacked,default]" ) {
    const size_type size = GENERATE(2, 4, 8, 12, 16);
    test_stacked_transpose<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "LinAlg: Stacked Transpose of axes that aren't last", "[LinAlg,transpose,stacked,dims]" ) {
    const size_type size = GENERATE(2, 4, 8, 12, 16);
    test_stacked_transpose<TestType>(size,0,1);
    test_stacked_transpose<TestType>(size,0,2);
}

FLOAT_NSL_TEST_CASE( "LinAlg: Transpose of Transpose", "[LinAlg,transpose]" ) {
    const size_type size = GENERATE(2, 4, 8, 12, 16);
    test_transpose_of_transpose<TestType>(size);
    test_transpose_of_transpose<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "LinAlg: Transpose of matmul", "[LinAlg,transpose]" ) {
    const size_type size = GENERATE(2, 4, 8, 12, 16);
    test_transpose_matmul<TestType>(size);
    
}
