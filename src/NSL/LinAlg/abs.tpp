#ifndef NSL_LINALG_ABS_TPP
#define NSL_LINALG_ABS_TPP


#include <cmath>

#include "../Tensor.hpp"

namespace NSL::LinAlg {
// NOTE: Why call std for c10 manipulations?
// I know it looks wrong!
//
// c10 puts its real, imag, abs, arg, and norm into the std namespace.
// See https://github.com/pytorch/pytorch/blob/master/c10/util/complex.h
// (as recently as 085e2f7)

    //! Returns the real-type absolute value, regardless of whether the passed value is real or `complex<>`.
template<NSL::Concept::isNumber Type>
NSL::RealTypeOf<Type> abs(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::abs(value);
    }
    else {
        return std::abs(value);
    }
}

template<typename Type>
inline NSL::Tensor<NSL::RealTypeOf<Type>> abs(const NSL::Tensor<Type> &T){
    // preform a deep copy of the tensor;
    NSL::Tensor<Type> Tcopy(T,true);
    return Tcopy.abs();
}

} // namespace NSL::LinAlg
#endif //NSL_LINALG_ABS_HPP
