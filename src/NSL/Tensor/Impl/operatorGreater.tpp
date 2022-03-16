#ifndef NSL_TENSOR_IMPL_OPERATOR_GREATER_TPP
#define NSL_TENSOR_IMPL_OPERATOR_GREATER_TPP

#include "../tensor.hpp"

namespace NSL{

template<typename SelfType,typename OtherType>
inline NSL::Tensor<bool> operator>(const NSL::Tensor<SelfType> & self, const NSL::Tensor<OtherType> & other) {
    return torch::Tensor(self) > torch::Tensor(other);
}

template<NSL::Concept::isNumber ValueType,typename TensorType>
inline NSL::Tensor<bool> operator>(const NSL::Tensor<TensorType> & tensor, const ValueType & value) {
    return torch::Tensor(tensor) > value;
}

template<NSL::Concept::isNumber ValueType,typename TensorType>
inline NSL::Tensor<bool> operator>(const ValueType & value, const NSL::Tensor<TensorType> & tensor) {
    // calls above
    return tensor > value;
}

} // namespace NSL

#endif // NSL_TENSOR_IMPL_OPERATOR_GREATER_TPP

