#include "../test.hpp"
#include "NSL.hpp"
#include <iostream>

using size_type = int64_t;

/*!
 *  These two tests differ slightly in their logic ONLY because we couldn't get
 *  either to work properly!  In a world with a just god they would be the same,
 *  except one would have log.
 *
 *  We should aim to harmonize them.
 *  However, we have a real loss-of-precision problem when using floats
 *  (doubles fair better).
 *
 * todo: a real fix requires addressing #43
 *
 **/


template<typename T>
void test_det(const size_type size) {
    
    //setting precision
    auto limit =  100*std::numeric_limits<T>::epsilon();
    
    NSL::Tensor<T> A(size,size);
    A.rand();
    // Make sure the spectrum of A is bound from below.
    A = A + NSL::LinAlg::adjoint(A);
    auto B = NSL::LinAlg::mat_exp(A);//A.mat_exp();
    A = B;

    //Testing if det(a_dagger) = det(a)*
    T det = NSL::LinAlg::det(A);
    T detstar = NSL::LinAlg::det(NSL::LinAlg::adjoint(A));

    INFO( det )
    INFO( detstar )
    REQUIRE( std::abs(NSL::real(det) - NSL::real(detstar)) <= limit );
    REQUIRE( std::abs(NSL::imag(det) + NSL::imag(detstar)) <= limit );

}

template<typename T>
void test_logdet(const size_type size) {

    //setting precision
    auto limit =  100*std::numeric_limits<T>::epsilon();

    NSL::Tensor<T> A(size,size);
    A.rand();
    // Make sure the spectrum of A is bound from below.
    A = A + NSL::LinAlg::adjoint(A);
    auto B = NSL::LinAlg::mat_exp(A);//A.mat_exp();
    A = B;

    //Testing if logdet(a_dagger) = logdet(a)*
    T det = NSL::LinAlg::logdet(A);
    T detstar = NSL::LinAlg::logdet(NSL::LinAlg::adjoint(A));

    INFO( det )
    INFO( detstar )
    REQUIRE( std::abs(1. - NSL::real(det)/NSL::real(detstar)) <= limit );
    REQUIRE( std::abs(1. + NSL::imag(det)/NSL::imag(detstar)) <= limit );
//    REQUIRE( std::abs(NSL::real(det) - NSL::real(detstar)) <= limit );
//    REQUIRE( std::abs(NSL::imag(det) + NSL::imag(detstar)) <= limit );
    

}

//Test Cases

FLOAT_NSL_TEST_CASE( "LinAlg: det", "[LinAlg, det]" ) {
    
    const size_type size = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_det<TestType>(size);
    
}

FLOAT_NSL_TEST_CASE( "LinAlg: logdet", "[LinAlg, logdet]" ) {
    
    const size_type size = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logdet<TestType>(size);
    
}

//! todo: add tests of stacked determinants.  See issue #42.
//! todo: understand why 100*epsilon is the right limit.
/*! todo: torch seems to have a problem with float32; see issue #43.
 *  For example, in python try
 *  the below copy-and-pastable code:
A = torch.rand((12,12), dtype=torch.float32)
B = A + A.conj().transpose(0,1)
C = torch.matrix_exp(B)
C.logdet() - B.trace()
 *  which can give something like 
 *  tensor(0.0147)!
 **/
