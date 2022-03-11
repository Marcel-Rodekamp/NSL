#ifndef NSL_TEST_HPP
#define NSL_TEST_HPP

// for CATCH2::INFO string computations
#include<string>

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

#endif
