#include "../test.hpp"
#include "NSL.hpp"

// Here we test exp, cos, sin, tan, cosh, sinh, tanh with a variety of trigonometric identities.

template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_pythagoras(SizeTypes ... Ns);

template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_pythagoras_tan(SizeTypes ... Ns);

template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_pythagoras_sec_csc(SizeTypes ... Ns);

template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_half_periods(SizeTypes ... Ns);

template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_euler(SizeTypes ... Ns);

template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_trig_hyperbolic_relations(SizeTypes ... Ns);

template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_parity(SizeTypes ... Ns);

//=======================================================================
// TEST CASES
//=======================================================================

COMPLEX_NSL_TEST_CASE("Tensor exp(0) = 1", "[Tensor,exp-of-zero-is-one]"){
    NSL::Tensor<TestType> A(1,2,4);
    REQUIRE( almost_equal(A.exp(), TestType(1)).all() );
}

COMPLEX_NSL_TEST_CASE("Tensor sin^2+cos^2=1", "[Tensor,pythagoras]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size2 = GENERATE(1,2,4,8,16,32);
    test_pythagoras<TestType>(size0,size1,size2);
}

COMPLEX_NSL_TEST_CASE("Tensor tan^2+1=sec^2", "[Tensor,pythagoras_tan]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size2 = GENERATE(1,2,4,8,16,32);
    test_pythagoras_tan<TestType>(size0,size1,size2);
}

COMPLEX_NSL_TEST_CASE("Tensor sec^2+csc^2=sec^2 csc^2", "[Tensor,pythagoras_sec_csc]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size2 = GENERATE(1,2,4,8,16,32);
    test_pythagoras_sec_csc<TestType>(size0,size1,size2);
}

COMPLEX_NSL_TEST_CASE("Tensor half-period shifts", "[Tensor,trig,half-period]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size2 = GENERATE(1,2,4,8,16,32);
    test_half_periods<TestType>(size0,size1,size2);
}

COMPLEX_NSL_TEST_CASE("Tensor euler identity", "[Tensor,trig,euler-identity]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size2 = GENERATE(1,2,4,8,16,32);
    test_euler<TestType>(size0,size1,size2);
}

COMPLEX_NSL_TEST_CASE("Tensor trig/hyperbolic", "[Tensor,trig,hyperbolic]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size2 = GENERATE(1,2,4,8,16,32);
    test_trig_hyperbolic_relations<TestType>(size0,size1,size2);
}

COMPLEX_NSL_TEST_CASE("Tensor trig parities", "[Tensor,trig,parity]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size2 = GENERATE(1,2,4,8,16,32);
    test_parity<TestType>(size0,size1,size2);
}

//=======================================================================
// Implementation Details: test_pythagoras
//=======================================================================
template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_pythagoras(SizeTypes ... Ns){
    // sin^2 + cos^2 = 1
    
    NSL::Tensor<Type> A(Ns...);A.rand();
    Type * addr_bak = A.data();
    NSL::Tensor<Type> B(A,true);
    Type * bddr_bak = B.data();

    A.sin();
    B.cos();

    REQUIRE( addr_bak == A.data() );
    REQUIRE( bddr_bak == B.data() );
    REQUIRE( almost_equal(A*A+B*B, Type(1.)).all() );
}

//=======================================================================
// Implementation Details: test_pythagoras_tan
//=======================================================================
template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_pythagoras_tan(SizeTypes ... Ns){
    // tan^2 + 1 = sec^2
    
    NSL::Tensor<Type> A(Ns...);A.rand();
    Type * addr_bak = A.data();
    NSL::Tensor<Type> B(A,true);
    Type * bddr_bak = B.data();

    A.cos();
    B.tan();

    REQUIRE( addr_bak == A.data() );
    REQUIRE( bddr_bak == B.data() );
    REQUIRE( almost_equal( B*B + 1, 1/(A*A) ).all() );
}

//=======================================================================
// Implementation Details: test_pythagoras_sec_csc
//=======================================================================
template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_pythagoras_sec_csc(SizeTypes ... Ns){
    // sec^2 + csc^2 = sec^2 csc^2
    
    NSL::Tensor<Type> A(Ns...);A.rand();
    Type * addr_bak = A.data();
    NSL::Tensor<Type> B(A,true);
    Type * bddr_bak = B.data();

    A.cos();
    B.sin();

    REQUIRE( addr_bak == A.data() );
    REQUIRE( bddr_bak == B.data() );
    REQUIRE( almost_equal( 1/(B*B) + 1/(A*A), 1/(B*B*A*A) ).all() );
}

//=======================================================================
// Implementation Details: test_half_periods
//=======================================================================
template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_half_periods(SizeTypes ... Ns){
    NSL::Tensor<Type> A(Ns...);A.rand();
    Type * addr_bak = A.data();

    // sine
    {
        NSL::Tensor<Type> unshifted(A,true);
        NSL::Tensor<Type> shifted(A+std::numbers::pi_v<typename NSL::RT_extractor<Type>::type>,true);
        REQUIRE( almost_equal(unshifted.sin() + shifted.sin(), Type(0) ).all() );
    }

    // cosine
    {
        NSL::Tensor<Type> unshifted(A,true);
        NSL::Tensor<Type> shifted(A+std::numbers::pi_v<typename NSL::RT_extractor<Type>::type>,true);
        REQUIRE( almost_equal(unshifted.cos() + shifted.cos(), Type(0) ).all() );
    }

    // tan
    {
        NSL::Tensor<Type> unshifted(A,true);
        NSL::Tensor<Type> shifted(A+std::numbers::pi_v<typename NSL::RT_extractor<Type>::type>,true);
        REQUIRE( almost_equal(unshifted.tan(), shifted.tan()).all() );
    }

}

//=======================================================================
// Implementation Details: test_euler
//=======================================================================
template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_euler(SizeTypes ... Ns){
    NSL::Tensor<Type> A(Ns...);A.rand();
    Type * addr_bak = A.data();

    NSL::Tensor<Type> s(A, true);
    NSL::Tensor<Type> c(A, true);

    A*=Type(0,1);
    A.exp();

    REQUIRE( almost_equal(A, c.cos() + Type(0,1) * s.sin()).all() );

}

//=======================================================================
// Implementation Details: test_trig_hyperbolic_relations
//=======================================================================
template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_trig_hyperbolic_relations(SizeTypes ... Ns){
    NSL::Tensor<Type> A(Ns...);A.rand();

    // sinh x = -i sin ix
    {
        NSL::Tensor<Type> left(A,true);
        NSL::Tensor<Type> right(A*Type(0,1),true);
        REQUIRE( almost_equal(left.sinh() + Type(0,1) * right.sin(), Type(0) ).all() );
    }
    // cosh x = cos ix
    {
        NSL::Tensor<Type> left(A,true);
        NSL::Tensor<Type> right(A*Type(0,1),true);
        REQUIRE( almost_equal(left.cosh() - right.cos(), Type(0) ).all() );
    }
    // tanh x = -i tan ix
    {
        NSL::Tensor<Type> left(A,true);
        NSL::Tensor<Type> right(A*Type(0,1),true);
        REQUIRE( almost_equal(left.tanh() + Type(0,1) * right.tan(), Type(0) ).all() );
    }

}

//=======================================================================
// Implementation Details: test_parity
//=======================================================================
template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_parity(SizeTypes ... Ns){
    NSL::Tensor<Type> A(Ns...);A.rand();

    // sin(x) = -sin(-x)
    {
        NSL::Tensor<Type> left(A,true);
        NSL::Tensor<Type> right(A*Type(-1,0),true);
        REQUIRE( almost_equal(left.sin() + right.sin(), Type(0) ).all() );
    }
    // cos(x) = cos(-x)
    {
        NSL::Tensor<Type> left(A,true);
        NSL::Tensor<Type> right(A*Type(-1,0),true);
        REQUIRE( almost_equal(left.cos() - right.cos(), Type(0) ).all() );
    }
    // tan(x) = -tan(-x)
    {
        NSL::Tensor<Type> left(A,true);
        NSL::Tensor<Type> right(A*Type(-1,0),true);
        REQUIRE( almost_equal(left.tan() + right.tan(), Type(0) ).all() );
    }
    // sinh(x) = -sinh(-x)
    {
        NSL::Tensor<Type> left(A,true);
        NSL::Tensor<Type> right(A*Type(-1,0),true);
        REQUIRE( almost_equal(left.sinh() + right.sinh(), Type(0) ).all() );
    }
    // cosh(x) = cosh(-x)
    {
        NSL::Tensor<Type> left(A,true);
        NSL::Tensor<Type> right(A*Type(-1,0),true);
        REQUIRE( almost_equal(left.cosh() - right.cosh(), Type(0) ).all() );
    }
    // tanh(x) = -tanh(-x)
    {
        NSL::Tensor<Type> left(A,true);
        NSL::Tensor<Type> right(A*Type(-1,0),true);
        REQUIRE( almost_equal(left.tanh() + right.tanh(), Type(0) ).all() );
    }

}

