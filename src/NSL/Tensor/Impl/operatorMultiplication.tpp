#ifndef NSL_TENSOR_IMPL_OPERATOR_MULTIPLICATION_TPP
#define NSL_TENSOR_IMPL_OPERATOR_MULTIPLICATION_TPP

#include "../tensor.hpp"

namespace NSL{

//! \todo: Add proper casting

template<NSL::Concept::isNumber SelfType,NSL::Concept::isNumber OtherType>
inline NSL::Tensor<SelfType> operator*(const NSL::Tensor<SelfType> & self, const NSL::Tensor<OtherType> & other) {
    return torch::Tensor(self) * torch::Tensor(other);
}

template<NSL::Concept::isNumber ValueType,NSL::Concept::isNumber TensorType>
inline NSL::Tensor<TensorType> operator*(const NSL::Tensor<TensorType> & tensor, const ValueType & value) {
    return torch::Tensor(tensor) * value;
}

template<NSL::Concept::isNumber ValueType,NSL::Concept::isNumber TensorType>
inline NSL::Tensor<TensorType> operator*(const ValueType & value, const NSL::Tensor<TensorType> & tensor) {
    // calls above
    return tensor * value;
}

} // namespace NSL

#endif // NSL_TENSOR_IMPL_OPERATOR_MULTIPLICATION_TPP

