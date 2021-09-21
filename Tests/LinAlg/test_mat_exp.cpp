//#include <complex>
#include "complex.hpp"
#include "catch2/catch.hpp"
#include "LinAlg/mat_exp.hpp"
#include "LinAlg/exp.hpp"
#include "LinAlg/abs.hpp"
#include <typeinfo>

// Torch requirement
using size_type = long int;

template<typename T>
void test_exponential_of_zero(const size_type & size){
    // const std::size_t & size, number of elements
    // Initializes a 2D Tensor of type T
    // with size elements in each direction.
    // Checks if its exponential is the identity matrix.
    // Note: NSL::Tensor::data()
    // Note: Requires conversion from int to type T
    NSL::Tensor<T> Tr(size, size);

    NSL::Tensor<T> exponentiated = NSL::LinAlg::mat_exp(Tr);

    NSL::Tensor<T> one(size, size);
    for(int i = 0; i < size; ++i) {
        one(i,i) = 1.;
    }

    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            REQUIRE(exponentiated(i,j) == one(i,j));
        }
    }
}

template<typename T>
void test_exponential_of_diagonal(const size_type & size){
    // const std::size_t & size, number of elements
    // const T & exponent, entry to put on the diagonal
    // Initializes an identity matrix of type T
    // with size elements in each direction.
    // Checks if exponentiating the identity = an identity filled with exponentials.
    // Note: NSL::Tensor::data()
    // Note: Requires conversion from int to type T

    auto limit = std::pow(10, std::numeric_limits<T>::digits10);

    NSL::Tensor<T> exponent(size);
    exponent.rand();

    NSL::Tensor<T> expected(size, size);
    for(int i = 0; i < size; ++i) {
        expected(i,i) = NSL::LinAlg::exp(exponent(i));
    }

    NSL::Tensor<T> diagonal(size, size);
    for (int i = 0; i < size; ++i ) {
        diagonal(i,i) = exponent(i);
    }

    NSL::Tensor<T> exponentiated = NSL::LinAlg::mat_exp(diagonal);

    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            auto res = NSL::LinAlg::abs(exponentiated(i,j)-expected(i,j));
            INFO("Type = " << typeid(T).name() << ", Res = " << res << ", limit = " << limit);
            REQUIRE( res <= limit);
        }
    }
}

template<typename T>
void test_exponential_of_hermitian(const size_type & size){
    // Generate real eigenvalues
    NSL::Tensor<T> eigenvalue(size);
    eigenvalue.rand().real();

    NSL::Tensor<T> diagonal(size,size);
    for(int i = 0; i < size; ++i ) {
        diagonal(i,i) = eigenvalue(i);
    }

    // Construct a random unitary
    NSL::Tensor<T> U(size, size);
    U.rand();
    auto det = U.det();
    U /= pow(det, 1/size);

    // Construct the exponential matrix two ways
    NSL::Tensor<T> clever(size, size) = U * NSL::LinAlg::mat_exp( diagonal ) * U.adjoint();
    NSL::Tensor<T> brute(size,size)   = NSL::LinAlg::mat_exp( U * diagonal * U.adjoint() );

    // Compare
    auto limit = std::pow(10, std::numeric_limits<T>::digits10);
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            auto res = NSL::LinAlg::abs(clever(i,j)-brute(i,j));
            INFO("Type = " << typeid(T).name() << ", Res = " << res << ", limit = " << limit);
            REQUIRE( res <= limit);
        }
    }
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

TEST_CASE( "LinAlg: Mat_Exp of zero", "[LinAlg,mat_exp,zero]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);

    // floating point types
    test_exponential_of_zero<float>(size);
    test_exponential_of_zero<double>(size);
    test_exponential_of_zero<NSL::complex<float>>(size);
    test_exponential_of_zero<NSL::complex<double>>(size);
}

TEST_CASE( "LinAlg: Mat_Exp of diagonal", "[LinAlg,mat_exp,diagonal]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);

    // floating point types
    test_exponential_of_diagonal<float>(size);
    test_exponential_of_diagonal<double>(size);
    test_exponential_of_diagonal<NSL::complex<float>>(size);
    test_exponential_of_diagonal<NSL::complex<double>>(size);
}

TEST_CASE( "LinAlg: Mat_Exp of hermitian", "LinAlg,mat_exp,hermitian]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);

    // floating point types
    test_exponential_of_hermitian<float>(size);
    test_exponential_of_hermitian<double>(size);
    test_exponential_of_hermitian<NSL::complex<float>>(size);
    test_exponential_of_hermitian<NSL::complex<double>>(size);
}
