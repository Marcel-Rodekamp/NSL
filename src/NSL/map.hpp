#ifndef NANOSYSTEMLIBRARY_MAP_HPP
#define NANOSYSTEMLIBRARY_MAP_HPP

//! \file map.hpp

#include "complex.hpp"

namespace NSL {
//! Helper struct for generalizing `std::map` to complex valued keys
/*!
 * `std::map<Key,T,Compare,Allocator>` is imported with
 * ```
 * MapHelper<...>::mapType;
 * ```
 * */
template<
        typename Key,
        typename T,
        class Compare = std::less<Key>,
        class Allocator = std::allocator<std::pair<const Key, T>>
>
struct MapHelper {
    using mapType = std::map<Key, T, Compare, Allocator>;
};

//! Helper struct for generalizing `std::map` to complex valued keys
/*
 * The Lexicographical Ordering can be used as
 * ```
 * MapHelper<std::complex<double>,void>::LexicographicLess myLess;
 * bool b = myLess(std::complex<double>{0.,1.}, std::complex<double>{1.,0.});
 * ```
 *
*/
template<
        typename RealTypeKey,
        typename T
>
struct MapHelper<NSL::complex<RealTypeKey>, T> {
    struct LexicographicLess {
        bool operator()(NSL::complex<RealTypeKey> const &a, NSL::complex<RealTypeKey> const &b) const {
            return std::array<RealTypeKey, 2>{a.real(), a.imag()} < std::array<RealTypeKey, 2>{b.real(), b.imag()};
        }
    };

    using mapType = std::map<NSL::complex<RealTypeKey>, T, LexicographicLess>;
};

//! Generalization of `std::map` for complex valued Keys.
/*!
 * This alias is an import and extension for the `std::map`.
 * For all typenames `Key` which are not `NSL::complex`, `NSL::map`  is equal to
 * to `std::map` and has the same features as detailed here:
 * https://en.cppreference.com/w/cpp/container/map
 * if `Key` is `NSL::complex` the map is specialized to use a lexicographic
 * ordering on the key:
 * ```
 * lhs < rhs iff
 *      std::array{lhs.real(),lhs.imag()} < std::array{rhs.real(),rhs.imag()}
 * ```
 * which defines a weak order. For details see:
 * https://en.cppreference.com/w/cpp/algorithm/lexicographical_compare
 *
 * This differentiation is conveniently done via partial template specialization.
 * For non complex `Key`, `MapHelper<...>::mapType` simply imports `std::map`.
 * For `Key == NSL::complex<RealTypeKey>` the ordering is the lexicographical one.
 *
 * This is used e.g. in `NSL::Lattice` to store the computed exponential hopping
 * matrices together with the corresponding value of `delta` which in general
 * might be complex valued.
 * */
template<
        typename Key,
        typename T,
        class Compare = std::less<Key>,
        class Allocator = std::allocator<std::pair<const Key, T>>
>
using map = typename MapHelper<Key, T, Compare, Allocator>::mapType;
// With clang ^ this typename is necessary, although with g++11 it is not.
// Even with g++, before c++20 we would have to use:
//using map = typename MapHelper<Key,T>::mapType;

} //namespace NSL

#endif //NANOSYSTEMLIBRARY_MAP_HPP
