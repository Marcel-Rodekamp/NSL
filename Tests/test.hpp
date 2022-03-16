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



#endif
