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
    NSL::Tensor<Type> expand(const NSL::size_t & newSize) {
        std::vector<NSL::size_t> sizes = this->data_.sizes().vec();
        sizes.push_back(newSize);

        this->data_ = this->data_.unsqueeze(-1).expand(
            torch::IntArrayRef(sizes)
        ).clone();

        return NSL::Tensor<Type>(this);
    }

    // NSL::Tensor<Type> expand(const NSL::size_t & newSize, const NSL::size_t & newDim) {
    //     std::vector<NSL::size_t> sizes = this->data_.sizes().vec();
    //     sizes.push_back(newSize);

    //     this->data_ = this->data_.unsqueeze(newDim).expand(
    //         torch::IntArrayRef(sizes)
    //     ).clone();

    //     return NSL::Tensor<Type>(this);
    // }

    // Maybe there is a better way of doing this but this was quickly coded
    NSL::Tensor<Type> expand(const NSL::size_t & newSize, const NSL::size_t & newDim) {
        return this->expand(newSize).transpose(newDim, -1);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_EXPAND_TPP
