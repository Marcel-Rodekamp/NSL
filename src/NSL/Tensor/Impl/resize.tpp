#ifndef NSL_TENSOR_IMPL_RESIZE_TPP
#define NSL_TENSOR_IMPL_RESIZE_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorResize:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:

    //! Resize the Tensor given a shape vector
    NSL::Tensor<Type> resize(const std::vector<NSL::size_t> & shape){
        this->data_ = this->data_.resize_(torch::IntArrayRef(shape));

        return NSL::Tensor<Type>(this);
    }

    //! Resize the Tensor given the desired sizes
    NSL::Tensor<Type> resize(auto ... sizes) {
        return this->resize({sizes...});
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_RESIZE_TPP

