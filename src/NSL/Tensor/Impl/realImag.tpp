#ifndef NSL_TENSOR_IMPL_REAL_IMAG_TPP
#define NSL_TENSOR_IMPL_REAL_IMAG_TPP

#include "base.tpp"
#include "../Factory/like.tpp"

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
    NSL::Tensor<typename RT_extractor<Type>::type> real(){
        if constexpr(NSL::is_complex<Type>()){
            return NSL::Tensor<typename RT_extractor<Type>::type>(torch::real(this->data_));
        } else {
            return NSL::Tensor<Type>(this);
        }
    }

    NSL::Tensor<typename RT_extractor<Type>::type> real() const {
        if constexpr(NSL::is_complex<Type>()){
            return NSL::Tensor<typename RT_extractor<Type>::type>(torch::real(this->data_));
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
        Tensor<typename RT_extractor<Type>::type> imag(){
            if constexpr(NSL::is_complex<Type>()){
                return Tensor<typename RT_extractor<Type>::type>(torch::imag(this->data_));
            } else {
                return Tensor<typename RT_extractor<Type>::type>(torch::zeros_like(this->data_));
            }
        }

        Tensor<typename RT_extractor<Type>::type> imag() const {
            if constexpr(NSL::is_complex<Type>()){
                return Tensor<typename RT_extractor<Type>::type>(torch::imag(this->data_));
            } else {
                return Tensor<typename RT_extractor<Type>::type>(torch::zeros_like(this->data_));
            }
        }
};

} // namespace NSL::TensorImpl


namespace NSL{

//! If `complex<>`, returns the real part; otherwise returns the passed value.
template<typename Type>
NSL::Tensor<NSL::RealTypeOf<Type>> real(const NSL::Tensor<Type> &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return value.real();
    }
    else {
        return value;
    }
}

//! If `complex<>`, returns the imaginary part; otherwise returns 0.
template<typename Type>
NSL::Tensor<NSL::RealTypeOf<Type>> imag(const NSL::Tensor<Type> &value){
    if constexpr(is_complex<Type>()) {
        // See NOTE above for std::explanation.
        return value.imag();
    }
    else {
        return NSL::zeros_like(value);
    }
}

} // namespace NSL


#endif //NSL_TENSOR_IMPL_REAL_IMAG_TPP
