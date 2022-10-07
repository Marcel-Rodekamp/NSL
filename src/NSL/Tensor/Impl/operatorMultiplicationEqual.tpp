#ifndef NSL_TENSOR_IMPL_MULTIPLICATION_EQUAL_TPP
#define NSL_TENSOR_IMPL_MULTIPLICATION_EQUAL_TPP

#include "base.tpp"
#include "../../typePromotion.hpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorMultiplicationEqual:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:

    NSL::Tensor<Type> operator*=(const Type & other){
        this->data_*=other;
        return NSL::Tensor<Type>(this);
    }

    template<NSL::Concept::isNumber OtherType>
    NSL::Tensor<NSL::CommonTypeOf<Type,OtherType>> operator*=(const OtherType & other){
        this->data_.to(torch::TensorOptions().dtype<NSL::CommonTypeOf<Type,OtherType>>())*= static_cast<NSL::CommonTypeOf<Type,OtherType>>(other);
        return NSL::Tensor<NSL::CommonTypeOf<Type,OtherType>>(this);
    }

    NSL::Tensor<Type> operator*=(const NSL::Tensor<Type> & other){
        this->data_*=other;
        return NSL::Tensor<Type>(this);
    }

    template<NSL::Concept::isNumber OtherType>
    NSL::Tensor<NSL::CommonTypeOf<Type,OtherType>> operator*=(const NSL::Tensor<OtherType> & other){
        this->data_.to(torch::TensorOptions().dtype<NSL::CommonTypeOf<Type,OtherType>>())*= NSL::Tensor<NSL::CommonTypeOf<Type,OtherType>>(other); 
        return NSL::Tensor<NSL::CommonTypeOf<Type,OtherType>>(this);
    }
};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_MULTIPLICATION_EQUAL_TPP
