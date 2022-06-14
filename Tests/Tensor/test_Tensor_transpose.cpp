#include "../test.hpp"

// Torch requirement
using size_type = long int;

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_transpose(SizeTypes ... sizes){
    NSL::Tensor<Type> t(sizes...);
    t.rand();

    NSL::Tensor<Type> transpose = t.transpose();
    //Checking that the memory spaces are not disjunct
    REQUIRE( transpose.data() == t.data() );

}

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_transpose_of_transpose(SizeTypes ... sizes){
    NSL::Tensor<Type> t(sizes...);
    t.rand();

    NSL::Tensor<Type> doubletranspose = (t.transpose()).transpose();
    //Checking (A^T)^T = A
    REQUIRE( (doubletranspose == t).all() );

}

template<typename T>
void test_transpose_matmul(const size_type & size){
    NSL::Tensor<T> a(size,size),b(size,size), abtrans(size,size), btransatrans(size,size);
    a.rand();
    b.rand();
    abtrans = (NSL::LinAlg::mat_mul(a,b)).transpose();
    btransatrans = NSL::LinAlg::mat_mul(b.transpose(),a.transpose());
    //Checking (AB)^T = B^T A^T
    //Very low precision for double and float
    REQUIRE( almost_equal(abtrans, btransatrans, std::numeric_limits<T>::digits10-8).all() );

}
// =============================================================================
// Test Cases
// =============================================================================

FLOAT_NSL_TEST_CASE( "Tensor: Transpose1D", "[Tensor,1D,transpose]" ) {
    const size_type size = GENERATE(2, 4, 8, 12, 16);
    test_transpose<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "Tensor: Transpose2D", "[Tensor,2D,transpose]" ) {
    const size_type size_1 = GENERATE(2, 4, 8, 12, 16);
    const size_type size_2 = GENERATE(2, 4, 8, 12, 16);
    test_transpose<TestType>(size_1,size_2);
}

FLOAT_NSL_TEST_CASE( "Tensor: Transpose3D", "[Tensor,3D,transpose]" ) {
    const size_type size_1 = GENERATE(2, 4, 8, 12, 16);
    const size_type size_2 = GENERATE(2, 4, 8, 12, 16);
    const size_type size_3 = GENERATE(2, 4, 8, 12, 16);
    test_transpose<TestType>(size_1,size_2,size_3);
}

FLOAT_NSL_TEST_CASE( "Tensor: Transpose of Transpose 1D", "[Tensor,1D,transpose]" ) {
    const size_type size_1 = GENERATE(2, 4, 8, 12, 16);
    test_transpose_of_transpose<TestType>(size_1);
}

FLOAT_NSL_TEST_CASE( "Tensor: Transpose of Transpose 2D", "[Tensor,2D, transpose]" ) {
    const size_type size_1 = GENERATE(2, 4, 8, 12, 16);
    const size_type size_2 = GENERATE(2, 4, 8, 12, 16);
    test_transpose_of_transpose<TestType>(size_1,size_2);
}

FLOAT_NSL_TEST_CASE( "Tensor: Transpose of Transpose 3D", "[Tensor,3D, transpose]" ) {
    const size_type size_1 = GENERATE(2, 4, 8, 12, 16);
    const size_type size_2 = GENERATE(2, 4, 8, 12, 16);
    const size_type size_3 = GENERATE(2, 4, 8, 12, 16);
    test_transpose_of_transpose<TestType>(size_1,size_2,size_3);
}

FLOAT_NSL_TEST_CASE( "Tensor: Transpose of matmul", "[Tensor,transpose]" ) {
    const size_type size = GENERATE(2, 4, 8, 12, 16);
    test_transpose_matmul<TestType>(size);
    
}

