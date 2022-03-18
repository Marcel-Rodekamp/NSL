#ifndef NSL_TYPES_HPP
#define NSL_TYPES_HPP

namespace NSL {

//! Complex numbers
/*!
 *  Alias to the complex type of PyTorch `c10::complex<Type>`
 * */
template<typename Type>
using complex = c10::complex<Type>;

//! NSL's preferred indexing type
/*!
 * Negative indexing is allowed hence std::size_t is not feasible
 * As torch works with int64_t we follow their suggestion and apply the same
 * indexing default.
 * Using `NSL::size_t` also prevents narrowing warnings compared to `std::size_t`
 * */
using size_t = std::int64_t;

} // namespace NSL

#endif