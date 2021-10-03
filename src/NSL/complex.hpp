#ifndef NANOSYSTEMLIBRARY_COMPLEX_HPP
#define NANOSYSTEMLIBRARY_COMPLEX_HPP

//! \file complex.hpp

#include "torch/torch.h"

namespace NSL{

//! Complex numbers
/*!
 *  Alias to the complex type of PyTorch `c10::complex<Type>`
 * */
template<typename Type>
using complex = c10::complex<Type>;

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
    using value_type = T;
};

//! Helper struct to determine the real type of a `NSL::complex<RT>`
//! For details see `RT_extractor`
template<typename RT>
struct RT_extractor<NSL::complex<RT>> : public std::true_type {
    using value_type = RT;
};

//! Helper function to identify that a template parameter is of type `NSL::complex`
//! This function returns `RT_extractor<T>::value`. For details see `RT_extractor`
template<typename T>
constexpr bool is_complex(){
    return RT_extractor<T>::value;
}

}

#endif //NANOSYSTEMLIBRARY_COMPLEX_HPP