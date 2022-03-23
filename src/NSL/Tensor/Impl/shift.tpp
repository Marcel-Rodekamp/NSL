#ifndef NSL_TENSOR_IMPL_SHIFT_TPP
#define NSL_TENSOR_IMPL_SHIFT_TPP

#include "base.tpp"

namespace NSL::TensorImpl{

template <NSL::Concept::isNumber Type>
class TensorShift:
    virtual private NSL::TensorImpl::TensorBase<Type>
{
    public:
    //! Shift the 0-th dimension by `|shift|` elements in `sgn(shift)` direction.
    NSL::Tensor<Type> shift(const NSL::size_t & shift){
        this->data_ = this->data_.roll(shift,0);
        return NSL::Tensor<Type>(this);
    }

    //! Shift the dim-th dimension by `|shift|` elements in `sgn(shift)` direction.
    NSL::Tensor<Type> shift(const NSL::size_t & shift, const NSL::size_t & dim){
        this->data_ = this->data_.roll(shift,dim);
        return NSL::Tensor<Type>(this);
    }

    //! Shift the 0-th dimension by `|shift|` elements in `sgn(shift)` direction and multiply boundary.
    NSL::Tensor<Type> shift(const NSL::size_t & shift, const Type & boundary){
        this->data_ = this->data_.roll(shift,0);

        if(shift>0){
            this->data_.slice(/*dim=*/0,/*start=*/0,/*end=*/shift,/*step=*/1)*=boundary;
        } else {
            this->data_.slice(/*dim=*/0,/*start=*/this->shape(0)-shift,/*end=*/this->shape(0),/*step=*/1)*=boundary;
        }

        return NSL::Tensor<Type>(this);
    }

    //! Shift the dim-th dimension by `|shift|` elements in `sgn(shift)` direction and multiply boundary.
    NSL::Tensor<Type> shift(const NSL::size_t & shift, const NSL::size_t & dim, const Type &boundary){
        this->data_ = this->data_.roll(shift,dim);

        if(shift>0){
            this->data_.slice(/*dim=*/dim,/*start=*/0,/*end=*/shift,/*step=*/1)*=boundary;
        } else {
            this->data_.slice(/*dim=*/dim,/*start=*/this->shape(dim)-shift,/*end=*/this->shape(dim),/*step=*/1)*=boundary;
        }

        return NSL::Tensor<Type>(this);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_SHIFT_TPP
