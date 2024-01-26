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
    /* 
     *  ```python
     *  import numpy as np
     *  v = np.arange(0,10)
     *  v_shift_plus  = np.roll(v, +1)
     *  v_shift_minus = np.roll(v, -1)
     *  print(f'{v             = }')
     *  print(f'{v_shift_plus  = }')
     *  print(f'{v_shift_minus = }')
     *  ```
     *  The point of the above code is to show that if you start with a vector $v_i$ in your formula,
     *  and you manipulate the indices with (for example, mat vec) and wind up with $v_{i-1}$,
     *  then the thing you probably want in your code is to shift by PLUS ONE!  In other words,
     *  ```c++
     *  NSL::Tensor<int> v_i = NSL::arange(0,10);
     *  auto v_i_minus_1 = NSL::LinAlg::shift(v_i, +1);     // should return the same as v_shift_plus in the above python.
     *  ```
     */
    NSL::Tensor<Type> shift(NSL::size_t shift){
        this->data_ = this->data_.roll(shift,0);
        return NSL::Tensor<Type>(this);
    }

    //! Shift the dim-th dimension by `|shift|` elements in `sgn(shift)` direction.
    NSL::Tensor<Type> shift(NSL::size_t shift, NSL::size_t dim){
        this->data_ = this->data_.roll(shift,dim);
        return NSL::Tensor<Type>(this);
    }

    //! Shift the 0-th dimension by `|shift|` elements in `sgn(shift)` direction and multiply boundary.
    NSL::Tensor<Type> shift(NSL::size_t shift, const Type & boundary){
        this->data_ = this->data_.roll(shift,0);

        NSL::size_t N = NSL::Tensor<Type>(this).shape(0);

        if(shift>0){
            this->data_.slice(/*dim=*/0,/*start=*/0,/*end=*/shift,/*step=*/1)*=boundary;
        } else {
            this->data_.slice(/*dim=*/0,/*start=*/N-shift,/*end=*/N,/*step=*/1)*=boundary;
        }

        return NSL::Tensor<Type>(this);
    }

    //! Shift the dim-th dimension by `|shift|` elements in `sgn(shift)` direction and multiply boundary.
    NSL::Tensor<Type> shift(NSL::size_t shift, NSL::size_t dim, const Type &boundary){
        this->data_ = this->data_.roll(shift,dim);


        if(shift > 0){
            this->data_.slice(/*dim=*/dim,/*start=*/0,/*end=*/shift,/*step=*/1)*=boundary;
        } else if( shift < 0 ){
            NSL::size_t N = at::size(this->data_,dim);
            this->data_.slice(dim,N+shift,N) *= boundary;
        } 

        return NSL::Tensor<Type>(this);
    }

};

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_SHIFT_TPP
