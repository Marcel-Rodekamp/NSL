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

    //! Expanding the Tensor by adding a dimension of size `newSize` at the specified `dim`
    NSL::Tensor<Type> expand(const NSL::size_t &newSize, const int &dim) {
        std::vector<NSL::size_t> sizes = this->data_.sizes().vec();

        // Insert newSize at the chosen dimension position.
        sizes.insert(sizes.begin() + dim, newSize);

        // Unsqueeze at the chosen dimension instead of always at the end.
        this->data_ = this->data_.unsqueeze(dim).expand(
            torch::IntArrayRef(sizes)
        ).clone();

        return NSL::Tensor<Type>(this);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_EXPAND_TPP
