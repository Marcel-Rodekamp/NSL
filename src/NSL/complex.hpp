#ifndef NSL_COMPLEX_HPP
#define NSL_COMPLEX_HPP

//! \file complex.hpp

#include "torch/torch.h"
#include <complex>  // For std::complex type   abs
#include <cmath>    // For floating-point type abs
#include <cstdlib>  // For integral types      abs
#include <numbers>
#include <type_traits>

#include "types.hpp"

namespace NSL{
//! Helper struct to determine the real type of a `NSL::complex<RT>`
/*!
 * This structs can be used to identify `NSL::complex<RT>` template arguments and
 * extract the real type at compile time.
 * Constructing `RT_extractor<T>` can take two routes:
 *
 * 1. if `T` __is__ `NSL::complex<RT>` (for some real type `RT` e.g. `float`, `double`,...):
 *      * is_complex bool is true  : `RT_extractor<NSL::complex<RT>>::value = true`
 *      * extracted real type is `RT`: `RT_extractor<NSL::complex<RT>>::value_type = RT`
 * 2. if `T` __is not__ `NSL::complex<RT>` (for some real type `RT` e.g. `float`, `double`,...):
 *      * is_complex bool is false : `RT_extractor<T>::value = false`
 *      * extracted real type is `T` : `RT_extractor<T>::value_type = T`
 *
 * This is possible as  the `C++11` (or above) standard suggests that the more
 * specialized template is preferred over the less specialized one.\n
 * \n
 *
 * Example:\n
 * ```
 * template<typename T, typename RT = RT_extractor<T>()::value_type>
 * void f(){
 *      // Real Type is stored in RT
 *      // if T is not complex RT = T
 *
 *      // T is complex if:
 *      bool is_complex_1 = RT_extractor<T>()::value
 *
 *      // T is complex if:
 *      bool is_complex_2 = std::is_same<T,RT>()::value
 * }
 * ```
 * */
template<typename T>
struct RT_extractor : public std::false_type {
    // lagacy:
    //! \todo: remove value_type member
    using value_type = T;
    //! \todo: remove this old type
    using type = T;
};

//! Helper struct to determine the real type of a `NSL::complex<RT>`
//! For details see `RT_extractor`
template<typename RT>
struct RT_extractor<NSL::complex<RT>> : public std::true_type {
    // lagacy:
    //! \todo: remove value_type member
    using value_type = RT;
    //! \todo: remove this old type
    using type = RT;
};

template<typename T>
using RealTypeOf = typename RT_extractor<T>::type;

//! Helper function to identify that a template parameter is of type `NSL::complex`
//! This function returns `RT_extractor<T>::value`. For details see `RT_extractor`
template<typename T>
constexpr bool is_complex(){
    return RT_extractor<T>::value;
}

// NOTE: Why call std for c10 manipulations?
// I know it looks wrong!
//
// c10 puts its real, imag, abs, arg, and norm into the std namespace.
// See https://github.com/pytorch/pytorch/blob/master/c10/util/complex.h
// (as recently as 085e2f7)

//! If `complex<>`, returns the real part; otherwise returns the passed value.
template<typename Type>
NSL::RealTypeOf<Type> real(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::real(value);
    }
    else {
        return value;
    }
}

//! If `complex<>`, returns the imaginary part; otherwise returns 0.
template<typename Type>
NSL::RealTypeOf<Type> imag(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::imag(value);
    }
    else {
        return 0;
    }
}

}
#endif //NSL_COMPLEX_HPP
