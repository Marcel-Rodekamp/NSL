#ifndef NSL_TENSOR_IMPL_ABS_TPP
#define NSL_TENSOR_IMPL_ABS_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorAbs:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Elementwise absolute value
    /*!
     *
     * Whether a complex type or a real type, the absolute value is real.
     *
     * */
    NSL::Tensor<NSL::RealTypeOf<Type>> abs(){
        return torch::abs(this->data_);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_ABS_TPP
