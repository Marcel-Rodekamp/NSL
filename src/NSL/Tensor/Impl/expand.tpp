#ifndef NSL_TENSOR_IMPL_EXPAND_TPP
#define NSL_TENSOR_IMPL_EXPAND_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorExpand:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Expanding the Tensor by one dimension with size `newSize`
    NSL::Tensor<Type> & expand(const NSL::size_t & newSize) {
        // TODO: why does the compiler know what shape is?
        std::vector<NSL::size_t> sizes = this->shape();
        sizes.push_back(newSize);
        this->data_ = this->data_.unsqueeze(-1).expand(torch::IntArrayRef(sizes.data(),sizes.size()));
        return NSL::Tensor<Type>(this);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_EXPAND_TPP
