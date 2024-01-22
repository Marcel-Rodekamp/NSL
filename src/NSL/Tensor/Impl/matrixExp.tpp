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
     * The underlying implementation uses a Optimized Taylor Polynomial Approximation:
     * https://www.mdpi.com/2227-7390/7/12/1174
     *
     * https://discuss.pytorch.org/t/what-implementation-is-used-for-matrix-exp/159608
     *
     * use with care for large matrices.
     * */
    NSL::Tensor<Type> mat_exp() {
        this->data_ = this->data_.matrix_exp();
        return NSL::Tensor<Type>(this);
    }


};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_MATRIX_EXP_TPP
