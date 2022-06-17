#ifndef NSL_CONCEPTS_HPP
#define NSL_CONCEPTS_HPP

#include<concepts>
#include <type_traits>
#include "complex.hpp"
#include "sliceObj.tpp"

namespace NSL::Concept{

//! Ensuring Types of a parameter pack agree
/*!
 * C++ parameter packs can typically take arbitrary Types.
 * This [concept](https://en.cppreference.com/w/cpp/language/constraints) can be
 * used to ensure that all types are the same.
 *
 * Example:
 * ```
 * // can have all different types
 * template<typename ... Types>
 * f(Types ... args);
 *
 * // specify a single type for all arguments
 * template<Type T, isType<T> ... Types>
 * f(T arg0, Types ... args);
 * ```
 * The argument `arg0` is not required but helps when type deduction is wanted.
 *
 * */
template<typename T, typename U>
concept isType = requires(U u) {
    requires std::is_same<T, U>::value;
};

//! Ensuring Types of a parameter pack are integer
/*!
 * C++ parameter packs can typically take arbitrary Types.
 * This [concept](https://en.cppreference.com/w/cpp/language/constraints) can be
 * used to ensure that all types are the same.
 *
 * */
template<typename T>
concept isIntegral = requires(T t) {
    requires std::is_integral_v<T>;
};


//! Concept to check for complex numbers
template<typename T>
concept isComplex = requires(T t) {
    requires NSL::is_complex<T>();
};

//! Concept to check for numbers
/*!
 * The following types `T` are considered to be numbers
 *  - integral:
 *      - bool, char, char8_t, char16_t, char32_t, wchar_t, short, int, long, long long
 *  - floating point:
 *      - float, double, long double
 *  - complex:
 *      - NSL::complex
 * */
template<typename T>
concept isNumber = requires(T t) {
    requires isComplex<T> || std::is_floating_point_v<T> || std::is_integral_v<T>;
};

//! Concept to check for floating point numbers
/*!
 * The following types `T` are considered to be numbers
 *  - floating point:
 *      - float, double, long double
 *  - complex:
 *      - NSL::complex
 * */
template<typename T>
concept isFloatingPoint = requires(T t) {
    requires isComplex<T> || std::is_floating_point_v<T>;
};

//! Concept to check for integers or `NSL::Slice`
/*!
 * */
template<typename T>
concept isIndexer = requires(T t) {
    requires isIntegral<T> || std::is_same<T,NSL::Slice>::value;
};

//! Concept to check that `D` is derived from `B`
/*!
 * All kind of inheritances (public,protected,private) is allowed.
 * */
template<typename Base,typename Derived>
concept isDerived = requires(Base B,Derived D){
    requires std::is_base_of<Derived,Base>::value;
};

} // namespace NSL::Concept 

#endif //NSL_CONCEPTS_HPP
