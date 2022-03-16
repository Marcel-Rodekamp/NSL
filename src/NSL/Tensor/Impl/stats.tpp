#ifndef NSL_TENSOR_IMPL_STATS_TPP
#define NSL_TENSOR_IMPL_STATS_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorStats:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Get the extent of a certain dimension.
    /*!
     * Parameters:\n
     * * `const size_t & dim`: Dimension of which the extent should be queried.
     *
     * Behavior:\n
     * Returns the extent of the dimension specified by `dim`.
     * If no reallocation is performed the value will match the given parameter
     * to the constructor `NSL::Tensor<Type,RealType>::Tensor(Arg size0, SizeType... sizes)`.
     * */
    NSL::size_t shape(const NSL::size_t & dim) const {
        return this->data_.size(dim);
    }

    //! Get the extents of the tensor
    std::vector<NSL::size_t> shape() const {
        std::vector<NSL::size_t> out(this->data_.dim());
        torch::IntArrayRef shape = this->data_.sizes();
        std::copy(shape.begin(),shape.end(),out.begin());
        return out;
    }

    //! Get the dimension of the Tensor.
    /*!
     *  The dimension of the tensor is specified at construction by the number
     *  of integer arguments provided to the constructor `NSL::Tensor(Arg size0, SizeType... sizes)`
     * */
    NSL::size_t dim() const {
        return this->data_.dim();
    }

    //! get the total number of elements.
    NSL::size_t numel() const {
        return this->data_.numel();
    }

    //! Get the d-th stride of the tensor
    NSL::size_t strides(const NSL::size_t d) const {
        return this->data_.stride(d);
    }

    //! Get the strides of the tensor
    std::vector<NSL::size_t> strides() const {
        std::vector<NSL::size_t> out(this->data_.dim());
        torch::IntArrayRef shape = this->data_.strides();
        std::copy(shape.begin(),shape.end(),out.begin());
        return out;
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_STATS_TPP
