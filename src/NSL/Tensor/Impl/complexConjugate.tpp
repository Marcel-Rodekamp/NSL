#ifndef NSL_TENSOR_IMPL_COMPLEX_CONJUGATE_TPP
#define NSL_TENSOR_IMPL_COMPLEX_CONJUGATE_TPP

#include "base.tpp"
#include "types.hpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorComplexConj:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Complex Conjugation (Elementwise)
    NSL::Tensor<Type> conj() {

        if constexpr(NSL::is_complex<Type>()){
            //https://pytorch.org/docs/stable/generated/torch.conj.html?highlight=tensor%20conj_
            //conj returns a view with flipped bit.
            //
            // The natural way of implementing a "mutating conj" method for
            // a tensor could look something like this:
            // ```
            //     this->data_ = this->data_.conj();
            // ```
            // Unfortunately, this causes bugs as `torch::Tensor::conj` produces
            // a lazy view on the original tensor with the signbit of the 
            // imaginary part being flipped. 
            // This view somehow doesn't get copied using the assignement
            // `operator=` here. Therefore, we need to use the mutating
            // `torch::Tensor::copy_` which potentially explicitly copies
            // every value from the conjugated view into it's own memory
            // space, let's hope the compiler is smart enough to optimize it.
             
            this->data_ = torch::resolve_conj(
                torch::conj(this->data_)
            );
        }// else{
         // do nothing
         // }
        
        return NSL::Tensor<Type>(this);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_COMPLEX_CONJUGATE_TPP
