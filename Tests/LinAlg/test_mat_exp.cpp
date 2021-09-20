#include <complex>
#include "catch2/catch.hpp"
#include "LinAlg/mat_exp.hpp"
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
