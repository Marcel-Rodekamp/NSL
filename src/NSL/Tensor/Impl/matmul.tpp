#ifndef NSL_TENSOR_IMPL_MAT_MUL_TPP
#define NSL_TENSOR_IMPL_MAT_MUL_TPP

#include "base.tpp"
#include <stdexcept>

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
    auto mat_mul(const NSL::Tensor<OtherType> & other){
        // This is only a prelimary complex type casting
        if constexpr(NSL::is_complex<Type>() && !NSL::is_complex<OtherType>()){
            this->data_ = torch::matmul(this->data_,static_cast<NSL::Tensor<Type>>(other) );
            return NSL::Tensor<Type>(this);

        //! \todo: How to implement this in a save manner?
        //!        This compiles and if the returned tensor is used everything is fine.
        //!        Unfortunately, this now holds complex data but the Type template
        //!        is still real.
        } else if constexpr(!NSL::is_complex<Type>() && NSL::is_complex<OtherType>()){
            throw std::runtime_error("NSL::Tensor<real>::mat_mul(NSL::Tensor<complex>) can not be inplace.");
            //this->data_ = torch::matmul(static_cast<NSL::Tensor<OtherType>>(*this),other);
            //return std::move(NSL::Tensor<OtherType>(this));
        } else{
            this->data_ = torch::matmul(this->data_,other);
            return NSL::Tensor<Type>(this);
        }
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_MAT_MUL_TPP

