#ifndef NSL_TENSOR_IMPL_TRANSPOSE_TPP
#define NSL_TENSOR_IMPL_TRANSPOSE_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorTranspose:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Transpose dim0 dim1
    NSL::Tensor<Type> transpose(const NSL::size_t dim0, const NSL::size_t dim1) {
        this->data_.transpose_(dim0,dim1);
        return NSL::Tensor<Type>(this);
    }

    //! Matrix transpose
    NSL::Tensor<Type> transpose() {
        this->transpose(this->data_.dim()-1, this->data_.dim()-2);
        return NSL::Tensor<Type>(this);
    }

    //! Transpose dim0 dim1 creating an explicit copy
    NSL::Tensor<Type> T(const NSL::size_t dim0, const NSL::size_t dim1) {
        torch::Tensor data = torch::transpose(this->data_,dim0,dim1);
        return std::move(data);
    }

    //! Matrix transpose creating an explicit copy
    NSL::Tensor<Type> T() {
        torch::Tensor data = torch::transpose(this->data_,this->data_.dim()-1, this->data_.dim()-2);
        return std::move(data);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_TRANSPOSE_TPP
