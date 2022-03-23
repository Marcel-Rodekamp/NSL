#ifndef NSL_LINALG_SHIFT_TPP
#define NSL_LINALG_SHIFT_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! Shift the tensors dimension 0
template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> shift(NSL::Tensor<Type> & t, const NSL::size_t & shift){
    return NSL::Tensor<Type>(t,true).shift(shift);
}

//! Shift the tensors dimension `dim`
template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> shift(NSL::Tensor<Type> & t, const NSL::size_t & shift, const NSL::size_t & dim){
    return NSL::Tensor<Type>(t,true).shift(shift,dim);
}

//! Shift the tensors dimension 0 and apply `boundary` factor
template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> shift(NSL::Tensor<Type> & t, const NSL::size_t & shift, const Type & boundary){
    return NSL::Tensor<Type>(t,true).shift(shift,boundary);
}

//! Shift the tensors dimension `dim` and apply `boundary` factor
template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> shift(NSL::Tensor<Type> & t, const NSL::size_t & shift, const NSL::size_t & dim, const Type &boundary){
    return NSL::Tensor<Type>(t,true).shift(shift,dim,boundary);
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_SHIFT_TPP
