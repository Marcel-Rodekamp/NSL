#ifndef NSL_TENSOR_IMPL_TRIGONOMETRIC_TPP
#define NSL_TENSOR_IMPL_TRIGONOMETRIC_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorTrigonometric:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Elementwise exponential
    NSL::Tensor<Type> & exp() {
        this->data_.exp_();
        return NSL::Tensor<Type>(this);
    }

    //! Elementwise sine
    NSL::Tensor<Type> & sin() {
        this->data_.sin_();
        return NSL::Tensor<Type>(this);
    }

    //! Elementwise cosine
    NSL::Tensor<Type> & cos() {
        this->data_.cos_();
        return NSL::Tensor<Type>(this);
    }

    //! Elementwise tangent
    NSL::Tensor<Type> & tan() {
        this->data_.tan_();
        return NSL::Tensor<Type>(this);
    }

    //! Elementwise hyperbolic sine
    NSL::Tensor<Type> & sinh() {
        this->data_.sinh_();
        return NSL::Tensor<Type>(this);
    }

    //! Elementwise hyperbolic cosine
    NSL::Tensor<Type> & cosh() {
        this->data_.cosh_();
        return NSL::Tensor<Type>(this);
    }

    //! Elementwise hyperbolic tangent
    NSL::Tensor<Type> & tanh() {
        this->data_.tanh_();
        return NSL::Tensor<Type>(this);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_TRIGONOMETRIC_TPP
