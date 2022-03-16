#ifndef NSL_LINALG_ABS_HPP
#define NSL_LINALG_ABS_HPP


#include <cmath>
#include "../Tensor/tensor.hpp"
#include <torch/torch.h>

namespace NSL::LinAlg {

template<typename Type>
inline Type abs(Type number){
    return std::abs(number);
}

template<typename Type>
inline Type abs(NSL::complex<Type> number){
    return std::abs(number);
}

template<typename Type, typename RealType>
inline NSL::Tensor<RealType> abs(const NSL::Tensor<Type, RealType> &T){
    return NSL::Tensor<RealType>(T).abs();
}

} // namespace NSL::LinAlg
#endif //NSL_LINALG_ABS_HPP
