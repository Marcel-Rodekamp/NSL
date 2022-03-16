#ifndef NSL_TENSOR_IMPL_DIVISION_EQUAL_TPP
#define NSL_TENSOR_IMPL_DIVISION_EQUAL_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorDivisionEqual:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:

   NSL::Tensor<Type> operator/=(const NSL::Tensor<Type> & other){
        this->data_/=other;
        return Tensor<Type>(this);
    }

    template<NSL::Concept::isNumber OtherType>
    NSL::Tensor<Type> operator/=(const NSL::Tensor<OtherType> & other){
        this->data_/=other;
        return Tensor<Type>(this);
    }

    template<NSL::Concept::isNumber OtherType>
    NSL::Tensor<Type> operator/=(const OtherType & other){
        this->data_/=other;
        return Tensor<Type>(this);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_DIVISION_EQUAL_TPP
