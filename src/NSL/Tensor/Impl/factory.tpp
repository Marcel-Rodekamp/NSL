#ifndef NSL_TENSOR_IMPL_FACTORY_TPP
#define NSL_TENSOR_IMPL_FACTORY_TPP

// Import TensorBase & Tensor (declaration)
#include "base.tpp"

namespace NSL::TensorImpl{

//! Implements different factories for `NSL::Tensor`
template <NSL::Concept::isNumber Type>
class TensorFactories:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:

    //! Fill the tensor with pseudo-random numbers
    /*!
    * Fills the Tensor with pseudo-random numbers from the uniform distribution
    * \todo Generalize for different distributions
    */
    NSL::Tensor<Type> rand(){
        this->data_.uniform_();
        return NSL::Tensor<Type>(this);
    }

    //! Fill the tensor with pseudo-random numbers
    /*!
    * Fills the Tensor with pseudo-random numbers from the normal[mean = 0, variance = 1] distribution
    * \todo Generalize for different distributions
    */
    NSL::Tensor<Type> randn(){
        this->data_.normal_(0.0,1.414213562373095048801689);
        return NSL::Tensor<Type>(this);
    }


    //! Fill the tensor with pseudo-random integer numbers
    /*!
    * Fills the Tensor with pseudo-random integer numbers from the uniform[low,high] distribution
    * \todo Generalize for different distributions
    */
    NSL::Tensor<Type> randint(NSL::size_t low, NSL::size_t high){
        this->data_ = torch::randint_like(this->data_,low,high);
        return NSL::Tensor<Type>(this);
    }

    //! Fill the tensor with pseudo-random integer numbers
    /*!
    * Fills the Tensor with pseudo-random integer numbers from the uniform[0,high] distribution
    * \todo Generalize for different distributions
    */
    NSL::Tensor<Type> randint(NSL::size_t high){
        this->data_ = torch::randint_like(this->data_,high);
        return NSL::Tensor<Type>(this);
    }
};

} // namespace NSL::TensorImpl

#endif // NSL_TENSOR_IMPL_FACTORY_TPP
