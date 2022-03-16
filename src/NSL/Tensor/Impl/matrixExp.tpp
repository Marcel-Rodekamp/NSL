#ifndef NSL_TENSOR_IMPL_MATRIX_EXP_TPP
#define NSL_TENSOR_IMPL_MATRIX_EXP_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorMatrixExp:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Matrix exponential.
    /*!
     * \todo: Add documentation
     * */
    NSL::Tensor<Type> mat_exp() {
        this->data_ = this->data_.matrix_exp();
        return NSL::Tensor<Type>(this);
    }


};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_MATRIX_EXP_TPP
