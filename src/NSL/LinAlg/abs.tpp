#ifndef NSL_LINALG_ABS_TPP
#define NSL_LINALG_ABS_TPP


#include <cmath>

#include "../Tensor.hpp"

namespace NSL::LinAlg {

    //! Returns the real-type absolute value, regardless of whether the passed value is real or `complex<>`.
template<NSL::Concept::isNumber Type>
typename NSL::RT_extractor<Type>::type abs(const Type &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return std::abs(value);
    }
    else {
        return std::abs(value);
    }
}

template<typename Type>
inline NSL::Tensor<typename RT_extractor<Type>::value_type> abs(const NSL::Tensor<Type> &T){
    // preform a deep copy of the tensor;
    NSL::Tensor<Type> Tcopy(T,true);
    return Tcopy.abs();
}

} // namespace NSL::LinAlg
#endif //NSL_LINALG_ABS_HPP
