#ifndef NSL_PARAM_PACK_HELPERS_HPP
#define NSL_PARAM_PACK_HELPERS_HPP

#include <iostream>
#include <concepts>
#include <sstream>
#include <array>

#include "concepts.hpp"

namespace NSL {

//! Print parameter pack
/*!
 * Prints a parameter pack formatted to std::cout.
 * A new line is appended.
 * */
template<typename T0, typename ... T>
void print_args(T0 arg0, T ... args) {
    std::stringstream s;
    std::stringstream ss;
    ss << "(" << arg0;

    (
            ss << ... << (s << "," << args).rdbuf()
    );
    ss << ")";

    std::cout << ss.str() << std::endl;
}

//! Enumerate a given parameter pack
/*!
 *
 * */
template<typename ... Types>
constexpr std::array<int, sizeof...(Types)> enumPack(Types ...args) {
    int j = -1;
    auto enumer = [&j](auto dummy) constexpr -> int {
        j++;
        return j;
    };
    return std::array<int, sizeof...(args)>{(enumer(args))...};
}

//! Check if a type is contained in parameter pack
template<typename T, typename ...Ts>
struct packContains{
    static constexpr bool value = (std::is_same<T, Ts>::value || ...);
};

//! Check if a type of Base is contained in parameter pack
template<typename T, typename ...Ts>
struct packContainsDerived{
    static constexpr bool value = (std::is_base_of_v<T,Ts> || ...);
};

//! Check if a type is contained in parameter pack
template<typename T, typename ...Ts>
struct packContainsConvertible{
    static constexpr bool value = (std::is_convertible<T, Ts>::value || ...);
};


} // namespace NSL

#endif // NSL_PARAMETERPACKHELPERS_HPP
