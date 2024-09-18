#ifndef NSL_TENSOR_IMPL_SQRT_TPP
#define NSL_TENSOR_IMPL_SQRT_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorSqrt:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Elementwise exponential
    NSL::Tensor<Type> sqrt() {
        this->data_.sqrt_();
        return NSL::Tensor<Type>(this);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_SQRT_TPP