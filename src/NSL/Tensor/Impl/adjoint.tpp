#ifndef NSL_TENSOR_IMPL_ADJOINT_TPP
#define NSL_TENSOR_IMPL_ADJOINT_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorAdjoint:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
        //! Adjoint (elementwise complex conjugate & transpose) of `dim0` and `dim1`
        NSL::Tensor<Type> adjoint(const NSL::size_t & dim0, const NSL::size_t & dim1) {
            this->data_.transpose_(dim0,dim1);
            this->data_ = this->data_.conj();
            return NSL::Tensor<Type>(this);
        }

        //! Matrix adjoint (elementwise complex conjugate & matrix transpose)
        NSL::Tensor<Type> adjoint() {
            this->data_.transpose_(this->data_.dim()-1,this->data_.dim()-2);
            this->data_ = this->data_.conj();
            return NSL::Tensor<Type>(this);
        }

        //! Adjoint of `dim0` and `dim1` creates an explicit copy (elementwise complex conjugate & matrix transpose)
        NSL::Tensor<Type> H(const NSL::size_t & dim0, const NSL::size_t & dim1) {
            torch::Tensor data = torch::transpose(this->data_,dim0,dim1);
            data = data.conj();
            return data;
        }

        //! Matrix adjoint creates an explicit copy (elementwise complex conjugate & matrix transpose)
        NSL::Tensor<Type> H() {
            torch::Tensor data = torch::transpose(this->data_,this->data_.dim()-1,this->data_.dim()-2);
            data = data.conj();
            return data;
        }
};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_ADJOINT_TPP
