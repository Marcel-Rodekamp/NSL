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
    // new way :)
    using type = T;
};

//! Helper struct to determine the real type of a `NSL::complex<RT>`
//! For details see `RT_extractor`
template<typename RT>
struct RT_extractor<NSL::complex<RT>> : public std::true_type {
    // lagacy:
    //! \todo: remove value_type member
    using value_type = RT;
    // new way :)
    using type = RT;
};

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
template<typename Type, typename RealType = typename NSL::RT_extractor<Type>::value_type>
RealType real(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::real(value);
    }
    else {
        return value;
    }
}

//! If `complex<>`, returns the imaginary part; otherwise returns 0.
template<typename Type, typename RealType = typename NSL::RT_extractor<Type>::value_type>
RealType imag(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::imag(value);
    }
    else {
        return 0;
    }
}

//! Returns the real-type absolute value, regardless of whether the passed value is real or `complex<>`.
template<typename Type, typename RealType = typename NSL::RT_extractor<Type>::value_type>
RealType abs(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::abs(value);
    }
    else {
        return std::abs(value);
    }
}

//! Returns the complex conjugate, maintaining type (`complex<>` if `complex<>`, not if not).
template<typename Type, typename RealType = typename NSL::RT_extractor<Type>::value_type>
inline Type conj(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::conj(value);
    }
    else {
        return value;
    }
}

//! Returns the complex conjugate, maintaining type (`complex<>` if `complex<>`, not if not).
template<typename Type, typename RealType = typename NSL::RT_extractor<Type>::value_type>
inline RealType arg(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::arg(value);
    }
    else {
        if(value > 0) return static_cast<RealType>(0);
        //! todo We should be very careful about the branch-cut of arg.
        // If we want arg to be single-valued, we should pick a finite interval,
        // say (-π,+π], which is 2π periodic.  However, the negative real axis
        // is right on the boundary.
        return static_cast<RealType>(+std::numbers::pi);
        // I picked + because in Mathematica
        //      Arg[-1] == +π
        // and I trust Wolfram to have these conventions sorted out.
        // Other choices are possible, and perhaps even are sensible!
        //! todo check what c10's std::arg does and ensure consistency.
    }
}

}

#endif //NSL_COMPLEX_HPP
