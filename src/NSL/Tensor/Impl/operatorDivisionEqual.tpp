#ifndef NSL_TENSOR_IMPL_DIVISION_EQUAL_TPP
#define NSL_TENSOR_IMPL_DIVISION_EQUAL_TPP

#include "base.tpp"
#include "../../typePromotion.hpp"
#include <c10/core/TensorOptions.h>

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorDivisionEqual:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:

    NSL::Tensor<NSL::CommonTypeOf<Type,float>> operator/=(const Type & other){
        this->data_.to(torch::TensorOptions().dtype<NSL::CommonTypeOf<Type,float>>() )/= static_cast<NSL::CommonTypeOf<Type,float>>(other);
        return NSL::Tensor<NSL::CommonTypeOf<Type,float>>(this);
    }

    template<NSL::Concept::isNumber OtherType>
    NSL::Tensor<NSL::CommonTypeOf< NSL::CommonTypeOf<Type,OtherType>, float >> operator/=(const OtherType & other){
        this->data_.to(torch::TensorOptions().dtype<NSL::CommonTypeOf< NSL::CommonTypeOf<Type,OtherType>, float >>())/= static_cast<NSL::CommonTypeOf< NSL::CommonTypeOf<Type,OtherType>, float >>(other);
        return NSL::Tensor<NSL::CommonTypeOf< NSL::CommonTypeOf<Type,OtherType>, float >>(this);
    }

    NSL::Tensor<NSL::CommonTypeOf<Type,float>> operator/=(const NSL::Tensor<Type> & other){
        this->data_.to(torch::TensorOptions().dtype<NSL::CommonTypeOf<Type,float>>() )/= NSL::Tensor<NSL::CommonTypeOf<Type,float>>(other);
        return NSL::Tensor<NSL::CommonTypeOf<Type,float>>(this);
    }

    template<NSL::Concept::isNumber OtherType>
    NSL::Tensor<NSL::CommonTypeOf< NSL::CommonTypeOf<Type,OtherType>, float >> operator/=(const NSL::Tensor<OtherType> & other){
        this->data_.to(torch::TensorOptions().dtype<NSL::CommonTypeOf< NSL::CommonTypeOf<Type,OtherType>, float >>())/= NSL::Tensor<NSL::CommonTypeOf< NSL::CommonTypeOf<Type,OtherType>, float >>(other); 
        return NSL::Tensor<NSL::CommonTypeOf< NSL::CommonTypeOf<Type,OtherType>, float >>(this);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_DIVISION_EQUAL_TPP
