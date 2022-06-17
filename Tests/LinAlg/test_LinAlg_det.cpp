#include "../test.hpp"
#include <iostream>

template<typename Type>
void test_det(const NSL::size_t & size);

template<typename Type>
void test_det_prod(const NSL::size_t & size);

template<typename Type>
void test_det_transpose(const NSL::size_t & size);

template<typename Type>
void test_det_exp_tr(const NSL::size_t & size);

// =============================================================================
// Test Cases
// =============================================================================

FLOAT_NSL_TEST_CASE( "LinAlg: determinant of identity", "[LinAlg,det,default]" ) {
    const NSL::size_t size = GENERATE(1, 100, 200, 500, 1000);
    test_det<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "LinAlg: determinant of transpose", "[LinAlg,detprod]" ) {
    const NSL::size_t size = GENERATE(1, 5, 10, 15, 20);
    test_det_prod<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "LinAlg: determinant of product", "[LinAlg,dettranspose]" ) {
    const NSL::size_t size = GENERATE(1, 5, 10, 15, 20);
    test_det_transpose<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "LinAlg: determinant and trace identity", "[LinAlg,detexptr]" ) {
    const NSL::size_t size = GENERATE(1, 5, 10, 15, 20);
    test_det_exp_tr<TestType>(size);
}


//=======================================================================
// Implementation Details
//=======================================================================

template<typename Type>
void test_det(const NSL::size_t & size){
    auto t = NSL::Matrix::Identity<Type>(size);
    
    auto det = NSL::LinAlg::det(t);

    REQUIRE( det == static_cast<Type>(1) );

}

template<typename Type>
void test_det_transpose(const NSL::size_t & size){
    auto A = NSL::Tensor<Type>(size, size).rand();
    auto AT = NSL::LinAlg::transpose(A);

    auto detA = NSL::LinAlg::det(A);
    auto detAT = NSL::LinAlg::det(AT);
    
    REQUIRE( almost_equal(detA, detAT, std::numeric_limits<Type>::digits10-5) );
    
}

template<typename Type>
void test_det_prod(const NSL::size_t & size){
    auto A = NSL::Tensor<Type>(size, size).rand();
    auto B = NSL::Tensor<Type>(size, size).rand();

    auto detA = NSL::LinAlg::det(A);
    auto detB = NSL::LinAlg::det(B);

    auto detAB = NSL::LinAlg::det(NSL::LinAlg::mat_mul(A, B));
    auto detBA = NSL::LinAlg::det(NSL::LinAlg::mat_mul(B, A));
    
    REQUIRE( almost_equal(detA * detB, detAB, std::numeric_limits<Type>::digits10-3) );
    REQUIRE( almost_equal(detA * detB, detBA, std::numeric_limits<Type>::digits10-3) );

}

template<typename Type>
void test_det_exp_tr(const NSL::size_t & size){
    auto A = NSL::Tensor<Type>(size, size).rand();

    auto expA = NSL::LinAlg::mat_exp(A);
    
    Type trA = 0;
    for(auto i = 0; i < size; ++i) trA += A(i,i);

    REQUIRE( almost_equal(NSL::LinAlg::det(expA), static_cast<Type>(NSL::LinAlg::exp(trA)), 2) );

}


