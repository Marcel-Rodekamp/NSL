#ifndef NSL_REAL_IMAG_TPP
#define NSL_REAL_IMAG_TPP

#include "complex.hpp"
#include "Tensor.hpp"
#include "Tensor/Factory/like.tpp"

namespace NSL{
// NOTE: Why call std for c10 manipulations?
// I know it looks wrong!
//
// c10 puts its real, imag, abs, arg, and norm into the std namespace.
// See https://github.com/pytorch/pytorch/blob/master/c10/util/complex.h
// (as recently as 085e2f7)

//! If `complex<>`, returns the real part; otherwise returns the passed value.
template<typename Type>
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
template<typename Type>
typename NSL::RT_extractor<Type>::type imag(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::imag(value);
    }
    else {
        return 0;
    }
}

template<typename Type>
NSL::Tensor<RealTypeOf<Type>> real(const NSL::Tensor<Type> & t){
    if constexpr(NSL::is_complex<Type>()){
        return NSL::Tensor<RealTypeOf<Type>>(
            torch::real(t)
        );
    } else {
        return t;
    }
}

template<typename Type>
NSL::Tensor<RealTypeOf<Type>> imag(const NSL::Tensor<Type> & t){
    if constexpr(NSL::is_complex<Type>()){
        return NSL::Tensor<RealTypeOf<Type>>(
            torch::imag(t)
        );
    } else {
        return NSL::zeros_like(t);
    }
}

} //namespace NSL
#endif // NSL_REAL_IMAG_TPP
