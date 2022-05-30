#ifndef NSL_LIN_ALG_COMPLEX_HPP
#define NSL_LIN_ALG_COMPLEX_HPP

#include "../complex.hpp"
#include "../concepts.hpp"


namespace NSL {
// NOTE: Why call std for c10 manipulations?
// I know it looks wrong!
//
// c10 puts its real, imag, abs, arg, and norm into the std namespace.
// See https://github.com/pytorch/pytorch/blob/master/c10/util/complex.h
// (as recently as 085e2f7)

//! If `complex<>`, returns the real part; otherwise returns the passed value.
template<NSL::Concept::isNumber Type>
typename NSL::RT_extractor<Type>::type real(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::real(value);
    }
    else {
        return value;
    }
}

//! If `complex<>`, returns the imaginary part; otherwise returns 0.
template<NSL::Concept::isNumber Type>
typename NSL::RT_extractor<Type>::type imag(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::imag(value);
    }
    else {
        return 0;
    }
}

} //namespace NSL

namespace NSL::LinAlg {

//! Returns the complex conjugate, maintaining type (`complex<>` if `complex<>`, not if not).
template<NSL::Concept::isNumber Type>
inline typename NSL::RT_extractor<Type>::type arg(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::arg(value);
    }
    else {
        if(value > 0) return static_cast<typename NSL::RT_extractor<Type>::type>(0);
        //! todo We should be very careful about the branch-cut of arg.
        // If we want arg to be single-valued, we should pick a finite interval,
        // say (-π,+π], which is 2π periodic.  However, the negative real axis
        // is right on the boundary.
        return static_cast<typename NSL::RT_extractor<Type>::type>(+std::numbers::pi);
        // I picked + because in Mathematica
        //      Arg[-1] == +π
        // and I trust Wolfram to have these conventions sorted out.
        // Other choices are possible, and perhaps even are sensible!
        //! todo check what c10's std::arg does and ensure consistency.
    }
}

} // namespace NSL::LinAlg

#endif //NSL_LIN_ALG_COMPLEX_HPP
