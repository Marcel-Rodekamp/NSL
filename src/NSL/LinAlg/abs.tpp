#ifndef NSL_LINALG_ABS_TPP
#define NSL_LINALG_ABS_TPP


#include <cmath>

#include "../Tensor.hpp"

namespace NSL::LinAlg {

template<typename Type>
inline Type abs(Type number){
    return std::abs(number);
}

template<typename Type>
inline Type abs(NSL::complex<Type> number){
    return std::abs(number);
}

template<typename Type>
inline NSL::Tensor<typename RT_extractor<Type>::value_type> abs(const NSL::Tensor<Type> &T){
    // preform a deep copy of the tensor;
    NSL::Tensor<Type> Tcopy(T,true);
    return Tcopy.abs();
}

} // namespace NSL::LinAlg
#endif //NSL_LINALG_ABS_HPP
