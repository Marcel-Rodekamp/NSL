#ifndef NSL_TENSOR_IMPL_REAL_IMAG_TPP
#define NSL_TENSOR_IMPL_REAL_IMAG_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorReal:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Elementwise real part
    /*!
     * If `Type` refers to `NSL::complex` return the real part of
     * each element of the tensor as `NSL::Tensor<RealType>`
     * Else `Type` refers to a RealType expression (e.g. `float`,`double`, ...)
     * and is simply returned.
     * */
    NSL::Tensor<NSL::RealTypeOf<Type>> real(){
        if constexpr(NSL::is_complex<Type>()){
            return NSL::Tensor<NSL::RealTypeOf<Type>>(torch::real(this->data_));
        } else {
            return NSL::Tensor<Type>(this);
        }
    }
};

template <NSL::Concept::isNumber Type>
class TensorImag:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    /*!
     * If `Type` refers to `NSL::complex` return the imaginary part of
     * each element of the tensor as `NSL::Tensor<RealType>`
     * Else `Type` refers to a RealType expression and does not have an imaginary
     * part, a Tensor with zeros is returned.
     * */
        Tensor<NSL::RealTypeOf<Type>> imag(){
            if constexpr(NSL::is_complex<Type>()){
                return Tensor<NSL::RealTypeOf<Type>>(torch::imag(this->data_));
            } else {
                return Tensor<NSL::RealTypeOf<Type>>(torch::zeros_like(this->data_));
            }
        }
};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_REAL_IMAG_TPP
