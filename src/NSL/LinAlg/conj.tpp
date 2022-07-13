#ifndef NSL_LINALG_CONJ_HPP
#define NSL_LINALG_CONJ_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {
// NOTE: Why call std for c10 manipulations?
// I know it looks wrong!
//
// c10 puts its real, imag, abs, arg, and norm into the std namespace.
// See https://github.com/pytorch/pytorch/blob/master/c10/util/complex.h
// (as recently as 085e2f7)

//! Returns the complex conjugate, maintaining type (`complex<>` if `complex<>`, not if not).
template<NSL::Concept::isNumber Type>
inline Type conj(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::conj(value);
    }
    else {
        return value;
    }
}

template <typename Type> 
inline NSL::Tensor<Type> conj(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(t,true).conj();
}

} // namespace NSL::LinAlg

#endif
