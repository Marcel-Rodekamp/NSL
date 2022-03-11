#ifndef NSL_LINALG_DET_HPP
#define NSL_LINALG_DET_HPP

#include "../Tensor/tensor.hpp"

namespace NSL {
namespace LinAlg {

//! Determinant of an n*n tensor or a stack of n*n tensors.
template <typename Type> 
Type det(const NSL::Tensor<Type> & t){
    //! todo: handle the case of stacked determinants; see issue #42.
    return to_torch(t).det().template item<Type>();
}

//! Log Determinant of an n*n tensor or a stack of n*n tensors.
template <typename Type> 
Type logdet(const NSL::Tensor<Type> & t){
    //! todo: handle the case of stacked determinants; see issue #42.
    return to_torch(t).det().log().template item<Type>();
}



} // namespace LinAlg
} // namespace NSL

#endif
