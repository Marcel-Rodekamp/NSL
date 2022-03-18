#ifndef NSL_TENSOR_IMPL_COMPLEX_CONJUGATE_TPP
#define NSL_TENSOR_IMPL_COMPLEX_CONJUGATE_TPP

#include "base.tpp"

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
            this->data_ = this->data_.conj();
        }// else{
         // do nothing
         // }
        
        return NSL::Tensor<Type>(this);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_COMPLEX_CONJUGATE_TPP
