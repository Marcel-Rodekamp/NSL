#include "../test.hpp"

// Torch requirement
using size_type = long int;

template<typename T>
void test_mat_mul_zero(const size_type & size);

template<typename T>
void test_mat_mul_one(const size_type & size);

template<typename T>
void test_mat_mul_inverse(const size_type & size);

template<typename T>
void test_stacked_mat_mul_zero(const size_type & size);

template<typename T>
void test_stacked_mat_mul_one(const size_type & size);

template<typename T>
void test_stacked_mat_mul_inverse(const size_type & size);


// =============================================================================
// Test Cases
// =============================================================================

FLOAT_NSL_TEST_CASE( "LinAlg: Matrix Multiplication -- zero", "[LinAlg,mat_mul,zero]" ) {
    const NSL::size_t size = GENERATE(1, 100, 200, 500, 1000);
    test_mat_mul_zero<TestType>(size);
}


FLOAT_NSL_TEST_CASE( "LinAlg: Matrix Multiplication -- one", "[LinAlg,mat_mul,one]" ) {
    const NSL::size_t size = GENERATE(1, 100, 200, 500, 1000);
    test_mat_mul_one<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "LinAlg: Matrix Multiplication -- inverses ", "[LinAlg,mat_mul,inverses]" ) {
    const NSL::size_t size = GENERATE(1, 100, 200, 500, 1000);
    test_mat_mul_inverse<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "LinAlg: Matrix Multiplication -- zero(stacked)", "[LinAlg,mat_mul,zero,stacked]" ) {
    const NSL::size_t size = GENERATE(1, 100, 200, 500, 1000);
    test_stacked_mat_mul_zero<TestType>(size);
}


FLOAT_NSL_TEST_CASE( "LinAlg: Matrix Multiplication -- one(stacked)", "[LinAlg,mat_mul,one,stacked]" ) {
    const NSL::size_t size = GENERATE(1, 100, 200, 500, 1000);
    test_stacked_mat_mul_one<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "LinAlg: Matrix Multiplication -- inverse(stacked)", "[LinAlg,mat_mul,inverses,stacked]" ) {
    const NSL::size_t size = GENERATE(1, 100, 200, 500, 1000);
    test_stacked_mat_mul_inverse<TestType>(size);
}

// =============================================================================
// Implementations
// =============================================================================


template<typename T>
void test_mat_mul_zero(const size_type & size){
    NSL::Tensor<T> t(size, size);
    t.rand();

    NSL::Tensor<T> zero_t(size, size);

    NSL::Tensor<T> zero_mul_left = NSL::LinAlg::mat_mul(zero_t, t);
    NSL::Tensor<T> zero_mul_right = NSL::LinAlg::mat_mul(t, zero_t);

    REQUIRE((zero_mul_left == zero_t).all() );
    REQUIRE((zero_mul_right == zero_t).all() );
}


template<typename T>
void test_mat_mul_one(const size_type & size){
    NSL::Tensor<T> t(size, size);
    t.rand();

    NSL::Tensor<T> one_t = NSL::Matrix::Identity<T>(size);

    NSL::Tensor<T> one_mul_left = NSL::LinAlg::mat_mul(one_t, t);
    NSL::Tensor<T> one_mul_right = NSL::LinAlg::mat_mul(t, one_t);

    REQUIRE((one_mul_left == t).all() );
    REQUIRE((one_mul_right == t).all() );
}


template<typename T>
void test_mat_mul_inverse(const size_type & size){
    NSL::Tensor<T> t(size, size);
    for(int i = 0; i < size; ++i) {
        t(i,i) = i+1;
    }

    NSL::Tensor<T> t_inv(size, size);
    for(int i = 0; i < size; ++i) {
        t_inv(i,i) = 1./(1+i);
    }

    auto one_t = NSL::Matrix::Identity<T>(size);
    NSL::Tensor<T> inv_mul_left = NSL::LinAlg::mat_mul(t_inv, t);
    NSL::Tensor<T> inv_mul_right = NSL::LinAlg::mat_mul(t, t_inv);

    REQUIRE(almost_equal(inv_mul_left, one_t).all() );
    REQUIRE(almost_equal(inv_mul_right, one_t).all() );
}



template<typename T>
void test_stacked_mat_mul_zero(const size_type & size){
    NSL::Tensor<T> t(10, size, size);
    t.rand();

    NSL::Tensor<T> zero_t(10,size, size);

    NSL::Tensor<T> zero_mul_left = NSL::LinAlg::mat_mul(zero_t, t);
    NSL::Tensor<T> zero_mul_right = NSL::LinAlg::mat_mul(t, zero_t);

    REQUIRE((zero_mul_left == zero_t).all() );
    REQUIRE((zero_mul_right == zero_t).all() );
}


template<typename T>
void test_stacked_mat_mul_one(const size_type & size){
    NSL::Tensor<T> t(10, size, size);
    t.rand();

    NSL::Tensor<T> one_t(10, size, size);
    for(int k = 0; k < 10; ++k) {
        one_t(k,NSL::Slice(), NSL::Slice()) = NSL::Matrix::Identity<T>(size);
    }

    NSL::Tensor<T> one_mul_left = NSL::LinAlg::mat_mul(one_t, t);
    NSL::Tensor<T> one_mul_right = NSL::LinAlg::mat_mul(t, one_t);

    REQUIRE((one_mul_left == t).all() );
    REQUIRE((one_mul_right == t).all() );
}


template<typename T>
void test_stacked_mat_mul_inverse(const size_type & size){
    NSL::Tensor<T> t(10, size, size);
    for(int k = 0; k < 10; ++k) {
        for(int i = 0; i < size; ++i) {
            t(k,i,i) = (i+1)*(k+1);
        }
    }

    NSL::Tensor<T> t_inv(10, size, size);
    for(int k = 0; k < 10; ++k) {
        for(int i = 0; i < size; ++i) {
            t_inv(k,i,i) = 1./(i+1)/(k+1);
        }
    }

    NSL::Tensor<T> one_t(10, size, size);
    for(int k = 0; k < 10; ++k) {
        for(int i = 0; i < size; ++i) {
            one_t(k,i,i) = 1.;
        }
    }
    NSL::Tensor<T> inv_mul_left = NSL::LinAlg::mat_mul(t_inv, t);
    NSL::Tensor<T> inv_mul_right = NSL::LinAlg::mat_mul(t, t_inv);

    REQUIRE(almost_equal(inv_mul_left, one_t).all() );
    REQUIRE(almost_equal(inv_mul_right, one_t).all() );
}


