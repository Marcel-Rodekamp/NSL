#ifndef NSL_TENSOR_IMPL_MAT_MUL_TPP
#define NSL_TENSOR_IMPL_MAT_MUL_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorMatmul:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    
    //! Matrix multiplication
    NSL::Tensor<Type> mat_mul(const NSL::Tensor<Type> & other){
        this->data_ = torch::matmul(this->data_,other);
        return NSL::Tensor<Type>(this);
    }

    //! Matrix multiplication
    /*!
     * \todo: Add propper type casting
     * */
    template<NSL::Concept::isNumber OtherType>
    NSL::Tensor<Type> mat_mul(const NSL::Tensor<OtherType> & other){
        this->data_ = torch::matmul(this->data_,other);
        return NSL::Tensor<Type>(this);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_MAT_MUL_TPP

