#ifndef NSL_TENSOR_IMPL_SLICE_TPP
#define NSL_TENSOR_IMPL_SLICE_TPP

#include "base.tpp"
#include "../../slice.hpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorSlice:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:

    //! Slice the tensor in 0th dimension
    /*!
     * Does the same as `operator()(NSL::Slice ...)` with only one Slice 
     * given.
     * */
    NSL::Tensor<Type> slice(NSL::Slice slice) {
        return this->operator()(slice);
    }
    
    //! Slice the Tensors `dim`th dimension from `start` to `end` with taking only every `step`th element.
    NSL::Tensor<Type> slice(const NSL::size_t & dim, const NSL::size_t & start, const NSL::size_t & end , const NSL::size_t & step = 1){
        torch::Tensor slice = this->data_.slice(dim,start,end,step);
        return std::move(slice);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_SLICE_TPP
