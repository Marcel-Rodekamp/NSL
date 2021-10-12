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

#define TORCH_TYPES int, FLOATING_POINT_TYPES, bool

// ====================================================================
// Custom TEST_CASE macros
// ====================================================================

#define TORCH_TYPE_TEST_CASE(_1, _2) TEMPLATE_TEST_CASE(_1, _2, TORCH_TYPES)

#endif
