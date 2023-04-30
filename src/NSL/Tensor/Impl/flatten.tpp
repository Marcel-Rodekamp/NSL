#ifndef NSL_TENSOR_IMPL_FLATTEN_TPP
#define NSL_TENSOR_IMPL_FLATTEN_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorFlatten:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    
    //! Flatten a tensor array
    NSL::Tensor<Type> flatten(){
        this->data_ = torch::flatten(this->data_);
        return NSL::Tensor<Type>(this);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_FLATTEN_TPP

