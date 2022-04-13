#ifndef NSL_TEST_HPP
#define NSL_TEST_HPP

// That's what we want to test
#include "NSL.hpp"

// for CATCH2::INFO string computations
#include<string>

// to compare numbers to machine prec
#include <limits>

// Using this to test
#include "catch2/catch.hpp"

// printing the types sometimes is helpful
#include <typeinfo>

// ====================================================================
// Type macros
// ====================================================================

#define REAL_TYPES float, double
#define COMPLEX_TYPES NSL::complex<float>, NSL::complex<double>
#define FLOATING_POINT_TYPES REAL_TYPES, COMPLEX_TYPES
#define INTEGER_TYPES int
#define NUMERIC_TYPES INTEGER_TYPES, FLOATING_POINT_TYPES
#define NSL_TYPES NUMERIC_TYPES, bool

// short int                  Not Supported by torch
// unsigned short int         Not Supported by torch
// unsigned int               Not Supported by torch
// size_type                  Not Supported by torch
// unsigned size_type         Not Supported by torch
// long size_type             Not Supported by torch
// unsigned long size_type    Not Supported by torch
// long double                Not Supported by torch
// NSL::complex<int>          Not Supported by torch

// ====================================================================
// Custom TEST_CASE macros
// ====================================================================

#define NSL_TEST_CASE(_1, _2) TEMPLATE_TEST_CASE(_1, _2, NSL_TYPES)
#define NUMERIC_NSL_TEST_CASE(_1, _2) TEMPLATE_TEST_CASE(_1, _2, NUMERIC_TYPES)
#define FLOAT_NSL_TEST_CASE(_1, _2) TEMPLATE_TEST_CASE(_1, _2, FLOATING_POINT_TYPES)
#define REAL_NSL_TEST_CASE(_1, _2) TEMPLATE_TEST_CASE(_1, _2, REAL_TYPES)
#define COMPLEX_NSL_TEST_CASE(_1, _2) TEMPLATE_TEST_CASE(_1, _2, COMPLEX_TYPES)
#define INTEGER_NSL_TEST_CASE(_1, _2) TEMPLATE_TEST_CASE(_1, _2, INTEGER_TYPES)
// todo: Hopefully we can drop the real-only test-cases.
// However, that requires solving issue #9.

//! compare two floating point numbers a,b
/*!
 *  \param a,b: Arguments which are compared
 *  \param factor: optional factor with which the machine precision can be scaled. Default is 1.
 *
 *  This is a relative compare. This means it checks for 
 *  \f[
 *      \left\vert 1 - \frac{a}{b} \right\vert \leq \text{factor} \cdot \epsilon<T>
 *  \f]
 *
 * */
template<NSL::Concept::isFloatingPoint T>
bool compare_floating_point(T a, T b, typename NSL::RT_extractor<T>::type factor = 10){
    return std::abs(static_cast<T>(1) - a/b) <= factor*std::numeric_limits<typename NSL::RT_extractor<T>::type>::epsilon();
}

//! compare two integer numbers a,b
/*!
 *  \param a,b: Arguments which are compared
 *  \param prec: optional precision to which the integers are compared. Default is 0.
 *
 *  This is an exact compare. This means it checks for 
 *  \f[
 *      \left\vert a - b \right\vert \leq \text{prec}
 *  \f]
 *
 * */
template<NSL::Concept::isIntegral Tint>
bool compare_integer(Tint a, Tint b, Tint prec = 0){
    if (prec == 0) {
        return a == b;
    } else {
        return std::abs(a-b) <= prec;
    }
}

//! Compare two floating point numbers up to numerical precision
/*!
 * Inspired by: https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
 * and        : https://stackoverflow.com/a/15012792
 *
 * In realtion to http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.22.6768
 * */
template<typename Type>
bool almost_equal(Type x, Type y, int ulp = std::numeric_limits<Type>::digits10)
{
    // Combining the relative magnitude of the inputs to express a relative tolerance.
    // Combining the relative with the absolute tolerance and testing for the "worst case"
    // allows the most sensible description of a finite precision error.
    Type max = std::max( {static_cast<Type>(1), std::fabs(x), std::fabs(y)} );

    // an additional factor provides the accuracy in digits and by default 
    // uses the default precision of the given Type
    return std::fabs(x-y) <= std::numeric_limits<Type>::epsilon() * max * ulp
        || std::fabs(x-y) < std::numeric_limits<Type>::min();
}

template<typename Type>
bool almost_equal(NSL::complex<Type> x, NSL::complex<Type> y, int ulp = std::numeric_limits<Type>::digits10)
{
    // Combining the relative magnitude of the inputs to express a relative tolerance.
    // Combining the relative with the absolute tolerance and testing for the "worst case"
    // allows the most sensible description of a finite precision error.
    Type maxReal = std::max( {static_cast<Type>(1), std::fabs(x.real()), std::fabs(y.real())} );
    Type maxImag = std::max( {static_cast<Type>(1), std::fabs(x.imag()), std::fabs(y.imag())} );

    // an additional factor provides the accuracy in digits and by default 
    // uses the default precision of the given Type
    // We further demand that both the real and imaginary part agree up to
    // the defined error tolerance.
    return (std::fabs(x.real()-y.real()) <= std::numeric_limits<Type>::epsilon() * maxReal * ulp 
            || std::fabs(x.real()-y.real()) < std::numeric_limits<Type>::min() ) 
        &&  
           ( std::fabs(x.imag()-y.imag()) <= std::numeric_limits<Type>::epsilon() * maxImag * ulp 
            || std::fabs(x.imag()-y.imag()) < std::numeric_limits<Type>::min() );
}

template<typename Type>
NSL::Tensor<bool> almost_equal(NSL::Tensor<Type> x, NSL::Tensor<Type> y, int ulp = std::numeric_limits<typename NSL::RT_extractor<Type>::value_type>::digits10){
    assertm( y.shape() == x.shape(), "To be almost equal two tensors must be the same shape.");

    NSL::Tensor<bool> result(static_cast<NSL::Tensor<typename NSL::RT_extractor<Type>::value_type>>(x));
    result = false;
    NSL::size_t elements = x.numel();
    for(NSL::size_t i = 0; i < elements; i++){
        result[i] = almost_equal(x[i], y[i], ulp);
    }

    return result;
}

#endif
