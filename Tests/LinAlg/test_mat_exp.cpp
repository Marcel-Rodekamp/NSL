//#include <complex>
#include "complex.hpp"
#include "../test.hpp"
#include "LinAlg/mat_exp.hpp"
#include "LinAlg/mat_mul.hpp"
#include "LinAlg/exp.hpp"
#include "LinAlg/det.hpp"
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

template<typename T, typename RT = typename NSL::RT_extractor<T>::value_type>
void test_exponential_of_hermitian(const size_type & size){
    INFO("Type = " << typeid(T).name());
    INFO("size = " << size);
    auto limit = std::pow(10, -std::numeric_limits<T>::digits10);

    // Construct an exponential of the form e^{H} where
    // H is Hermitian.

    // Any Hermitian matrix has real eigenvalues.
    NSL::Tensor<RT> eigenvalue(size);
    eigenvalue.rand();

    // To get a random H we can put the eigenvalues on the diagonal
    NSL::Tensor<T> diagonal(size,size);
    for(int i = 0; i < size; ++i ) {
        diagonal(i,i) = eigenvalue(i);
    }

    // and then conjugate with a random unitary matrix.
    NSL::Tensor<T> V(size, size);
    V.rand();
    T det = NSL::LinAlg::det(V);

    NSL::Tensor<T> U = V / std::pow(det, 1/size);
    T detU = NSL::LinAlg::det(U);
    INFO("det U = " << detU);

    REQUIRE(NSL::abs(detU - static_cast<T>(1)) < limit);

    INFO("DBUG0");

    auto Ubak = U;

    INFO("DBUG0.1");
    auto Udagger = Ubak.adjoint();

    INFO("DBUG1");

    // Since exp(UHU†) = U exp(H) U†, we expect
    NSL::Tensor<T> brute  = NSL::LinAlg::mat_exp( 
            NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(U , diagonal), Udagger) 
    );

    INFO("DBUG2");

    NSL::Tensor<T> clever = NSL::LinAlg::mat_mul(
            NSL::LinAlg::mat_mul(U, NSL::LinAlg::mat_exp( diagonal )), Udagger
    );

    INFO("DBUG3");

    // to be equal, up to some numerical precision.
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            auto res = NSL::abs(clever(i,j)-brute(i,j));
            INFO("Type = " << typeid(T).name() << ", Res = " << res << ", limit = " << limit);
            REQUIRE( res <= limit);
        }
    }

    // Moreover, since det(exp(UHU†)) = det(U exp(H) U†) = det( exp(H) )
    auto det_clever = NSL::LinAlg::det(clever);
    auto det_brute  = NSL::LinAlg::det(brute);

    // Finally, using the identity
    //      det e^{H} = e^tr H
    // we can compute the determinant directly.
    T trace = 0;
    for(int i = 0; i < size; ++i ) {
        trace += eigenvalue(i);
    }
    INFO("trace of eigenvalues = " << trace);
    auto determinant = exp(trace);

    INFO("target determinant   = " << determinant);
    INFO("clever determinant   = " << det_clever );
    INFO("brute  determinant   = " << det_brute  );
    // TODO: requires <= for NSL::Tensor?
    //REQUIRE( NSL::LinAlg::abs(determinant - det_clever) <= limit );
    //REQUIRE( NSL::LinAlg::abs(determinant - det_brute ) <= limit );
    //REQUIRE( NSL::LinAlg::abs(det_clever  - det_brute ) <= limit );

    // FIXME: in the mean time, fail
    REQUIRE( false );
}

// =============================================================================
// Test Cases
// =============================================================================

FLOAT_NSL_TEST_CASE( "LinAlg: Mat_Exp of zero", "[LinAlg,mat_exp,zero]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);

    test_exponential_of_zero<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "LinAlg: Mat_Exp of diagonal", "[LinAlg,mat_exp,diagonal]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);

    test_exponential_of_diagonal<TestType>(size);
}

FLOAT_NSL_TEST_CASE( "LinAlg: Mat_Exp of hermitian", "LinAlg,mat_exp,hermitian]" ) {
    const size_type size = GENERATE(1, 2, 4, 8, 16, 32, 64);

    test_exponential_of_hermitian<TestType>(size);
}
