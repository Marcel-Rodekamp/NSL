#ifndef NSL_TENSOR_IMPL_OPERATOR_DIVISION_TPP
#define NSL_TENSOR_IMPL_OPERATOR_DIVISION_TPP

#include "../../concepts.hpp"
#include "../../typePromotion.hpp"

namespace NSL{

// forward declare
template<NSL::Concept::isNumber Type> class Tensor;

template<NSL::Concept::isNumber SelfType, NSL::Concept::isNumber OtherType>
inline NSL::Tensor<
    NSL::CommonTypeOf<
        NSL::CommonTypeOf<SelfType,OtherType>,
        float
    >
> operator/(const NSL::Tensor<SelfType> & self, const NSL::Tensor<OtherType> & other) {
    return torch::Tensor(self).to(torch::TensorOptions().dtype<NSL::CommonTypeOf<SelfType,OtherType>>()) / torch::Tensor(other).to(torch::TensorOptions().dtype<NSL::CommonTypeOf<SelfType,OtherType>>());
}

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<NSL::CommonTypeOf<Type,float>> operator/(const NSL::Tensor<Type> & self, const NSL::Tensor<Type> & other) {
    return torch::Tensor(self) / torch::Tensor(other);
}

template<NSL::Concept::isNumber TensorType, NSL::Concept::isNumber ValueType>
inline NSL::Tensor<
    NSL::CommonTypeOf<
        NSL::CommonTypeOf<TensorType,ValueType>,
        float
    >
> operator/(const NSL::Tensor<TensorType> & tensor, const ValueType & value) {
    return torch::Tensor(tensor).to(torch::TensorOptions().dtype<NSL::CommonTypeOf<ValueType,TensorType>>()) / static_cast<NSL::CommonTypeOf<ValueType,TensorType>>(value);
}

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<NSL::CommonTypeOf<Type,float>> operator/(const NSL::Tensor<Type> & tensor, const Type & value) {
    return torch::Tensor(tensor) / value;
}

template<NSL::Concept::isNumber ValueType,NSL::Concept::isNumber TensorType>
inline NSL::Tensor<
    NSL::CommonTypeOf<
        NSL::CommonTypeOf<TensorType,ValueType>,
        float
    >
> operator/(const ValueType & value, const NSL::Tensor<TensorType> & tensor) {
    return static_cast<NSL::CommonTypeOf<NSL::CommonTypeOf<TensorType,ValueType>,float>>(value) / torch::Tensor(tensor).to(torch::TensorOptions().dtype<NSL::CommonTypeOf<NSL::CommonTypeOf<TensorType,ValueType>,float>>());
}

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<NSL::CommonTypeOf<Type,float>> operator/(const Type & value, const NSL::Tensor<Type> & tensor) {
    return static_cast<NSL::CommonTypeOf<Type,float>>(value) / torch::Tensor(tensor).to(torch::TensorOptions().dtype<NSL::CommonTypeOf<Type,float>>() ) ;
}

} // namespace NSL



#endif // NSL_TENSOR_IMPL_OPERATOR_DIVISION_TPP

