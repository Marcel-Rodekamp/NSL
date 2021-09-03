#include <complex>
#include "catch2/catch.hpp"
#include "Tensor/tensor.hpp"

void test_1D_constructor(const std::size_t & size){
    // const std::size_t & size, number of elements
    // Initializes a 1D Tensor of type
    // * float
    // with size elements and checks if it is initialized to 0.
    // Note: Requires NSL::Tensor::operator==(Type);

    NSL::Tensor<float> T_float(size);
    NSL::Tensor<double> T_double(size);
    NSL::Tensor<c10::complex<float>> T_cfloat(size);
    NSL::Tensor<c10::complex<double>> T_cdouble(size);

    REQUIRE(T_float == 0);
    REQUIRE(T_double == 0);
    REQUIRE(T_cfloat == c10::complex<float>(0,0));
    REQUIRE(T_cdouble == c10::complex<double>(0,0));
}

// =============================================================================
// Test Cases
// =============================================================================

TEST_CASE( "TENSOR: 1D Constructor", "[Tensor,Constructor,1D]" ) {
    const int size = GENERATE(1,100,200,500,1000);
    test_1D_constructor(size);
}
