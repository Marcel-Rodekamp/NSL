#ifndef NSL_TENSOR_IMPL_FACTORY_TPP
#define NSL_TENSOR_IMPL_FACTORY_TPP

// Import TensorBase & Tensor (declaration)
#include "base.tpp"
#include "RNG.tpp"

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
	torch::Tensor tmp = torch::zeros_like(this->data_);
	NSL::size_t num_elements = tmp.numel();
	NSL::Random<Type> rng;
	for (NSL::size_t i; i<num_elements; i++)
	    tmp.data_ptr<Type>()[i] = rng.uni_dis_rng(); 
        this->data_ = tmp.to(this->data_.device());
        return NSL::Tensor<Type>(this);
    }

    //! Fill the tensor with pseudo-random numbers
    /*!
    * Fills the Tensor with pseudo-random numbers from the normal[mean = 0, variance = 1] distribution
    * \todo Generalize for different distributions
    */
    NSL::Tensor<Type> randn(){
	torch::Tensor tmp = torch::zeros_like(this->data_);
	NSL::size_t num_elements = tmp.numel();
	NSL::Random<Type> rng;
	for (NSL::size_t i; i<num_elements; i++)
	    tmp.data_ptr<Type>()[i] = rng.nml_dis_rng(); 
        this->data_ = tmp.to(this->data_.device());
        return NSL::Tensor<Type>(this);
    }


    //! Fill the tensor with pseudo-random integer numbers
    /*!
    * Fills the Tensor with pseudo-random integer numbers from the uniform[low,high] distribution
    * \todo Generalize for different distributions
    */
    NSL::Tensor<Type> randint(NSL::size_t low, NSL::size_t high){
        //this->data_ = torch::randint_like(this->data_,low,high);
	torch::Tensor tmp = torch::zeros_like(this->data_);
	NSL::size_t num_elements = tmp.numel();
	NSL::Random<Type> rng;
	for (NSL::size_t i; i<num_elements; i++)
	    tmp.data_ptr<Type>()[i] = rng.uni_dis_lo_hi_rng(low, high); 
        this->data_ = tmp.to(this->data_.device());
        return NSL::Tensor<Type>(this);
    }

    //! Fill the tensor with pseudo-random integer numbers
    /*!
    * Fills the Tensor with pseudo-random integer numbers from the uniform[0,high] distribution
    * \todo Generalize for different distributions
    */
    NSL::Tensor<Type> randint(NSL::size_t high){
        //this->data_ = torch::randint_like(this->data_,high);
	torch::Tensor tmp = torch::zeros_like(this->data_);
	NSL::size_t num_elements = tmp.numel();
	NSL::Random<Type> rng;
	for (NSL::size_t i; i<num_elements; i++)
	    tmp.data_ptr<Type>()[i] = rng.uni_dis_hi_rng(high); 
        this->data_ = tmp.to(this->data_.device());
        return NSL::Tensor<Type>(this);
    }
};

} // namespace NSL::TensorImpl

#endif // NSL_TENSOR_IMPL_FACTORY_TPP
