#ifndef NSL_LINALG_CONJ_HPP
#define NSL_LINALG_CONJ_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

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
