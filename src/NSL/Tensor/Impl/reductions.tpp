#ifndef NSL_TENSOR_IMPL_REDUCTIONS_TPP
#define NSL_TENSOR_IMPL_REDUCTIONS_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorReductions:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Reduction: +
    NSL::Tensor<Type> sum(const NSL::size_t & dim){
        return this->data_.sum(dim);
    }

    //! Reduction: +
    NSL::Tensor<Type> tensor_sum(){
        return this->data_.sum();
    }

    //! Reduction: +
    Type sum(){
        return this->data_.sum().template item<Type>();
    }

    //! Reduction: *
    NSL::Tensor<Type> prod(const NSL::size_t & dim){
        return this->data_.prod(dim);
    }

    //! Reduction: *
    Type prod(){
        return this->data_.prod().template item<Type>();
    }

    //! Reduction: && (logical and)
    NSL::Tensor<Type> all(const NSL::size_t & dim){
        return this->data_.all(dim);
    }

    //! Reduction: && (logical and)
    Type all(){
        return this->data_.all().template item<Type>();
    }

    //! Reduction: || (logical or)
    NSL::Tensor<Type> any(const NSL::size_t & dim){
        return this->data_.any(int(dim));
    }

    //! Reduction: || (logical or)
    Type any(){
        return this->data_.any().template item<Type>();
    }

    //! Reduction: mean
    NSL::Tensor<Type> mean(const NSL::size_t & dim){
        return this->data_.mean(dim);
    }

    //! Reduction: mean
    Type mean(){
        return this->data_.mean().template item<Type>();
    }

    //! Reduction: var
    NSL::Tensor<Type> var(const NSL::size_t & dim){
        return this->data_.var( at::OptionalIntArrayRef(dim) );
    }

    //! Reduction: var
    Type var(){
        return this->data_.var().template item<Type>();
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_REDUCTIONS_TPP
