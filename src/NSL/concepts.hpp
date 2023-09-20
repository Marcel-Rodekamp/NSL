#ifndef NSL_CONCEPTS_HPP
#define NSL_CONCEPTS_HPP

#include<concepts>
#include <type_traits>
#include "complex.hpp"
#include "sliceObj.tpp"

// forward declare


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

//! Concept to check that `D` is derived from `B`
/*!
 * All kind of inheritances (public,protected,private) is allowed.
 * */
template<typename Derived,typename Base>
concept isDerived = std::is_base_of_v<Base,Derived>;

//! Concept to check for integers or `NSL::Slice`
/*!
 * */
template<typename T>
concept isIndexer = requires(T t) {
    requires isIntegral<T> || isDerived<T,NSL::Indexer>;
};

} // namespace NSL::Concept



// foward declare tensor 
namespace NSL {template<NSL::Concept::isNumber Type> class Tensor;}

namespace NSL::Concept {
// Hide this from the user
namespace {
template<typename T>
struct is_tensor: public std::true_type {};
template<NSL::Concept::isNumber T>
struct is_tensor<NSL::Tensor<T>>: public std::false_type {};
}

//! Concept to check if a type is a NSL::Tensor type
template<typename T>
concept isTensor = is_tensor<T>::value;

template <typename Derived, template <typename...> class Base>
struct isTemplateDerivedHelper {
    // We extract the template arguments of Derived 
    // which need to match the template argument of Base (assumption) 
    // Then compare the child structure of Base using the concept 'isDerived' 
    template <template<typename ...> class Derived_, typename... Args_>
        requires(isDerived<Derived_<Args_...>,Base<Args_...> >)
    constexpr static std::true_type checkConversionToBase(const Derived_<Args_...>*);
    
    // If it is just not a base class we return false
    constexpr static std::false_type checkConversionToBase(...);

    constexpr static bool value = decltype(
        isTemplateDerivedHelper<Derived,Base>::checkConversionToBase(std::declval<Derived*>())
    )::value;

};

//! Concept to check that a class is derived from a base class if the Base
//! and derived class are templated
template <typename Derived, template <typename...> class BaseClass>
concept isTemplateDerived = isTemplateDerivedHelper<Derived,BaseClass>::value;

} // namespace NSL::Concept 

#endif //NSL_CONCEPTS_HPP
