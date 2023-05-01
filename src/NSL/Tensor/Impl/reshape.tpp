#ifndef NSL_TENSOR_IMPL_RESHAPE_TPP
#define NSL_TENSOR_IMPL_RESHAPE_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorReshape:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:

    //! Reshaping the Tensor given a shape vector
    NSL::Tensor<Type> reshape(const std::vector<NSL::size_t> & shape){
        this->data_ = at::reshape(this->data_, torch::IntArrayRef(shape));

        return NSL::Tensor<Type>(this);
    }

    //! Reshaping the Tensor given the desired sizes
    NSL::Tensor<Type> reshape(auto ... sizes) {
        return this->reshape({sizes...});
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_RESHAPE_TPP

