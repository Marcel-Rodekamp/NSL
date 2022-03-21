#ifndef NSL_LINALG_MAT_VEC_HPP
#define NSL_LINALG_MAT_VEC_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg{

//! matrix times vector
/*!
 * Implementation is the same as `NSL::LinAlg::mat_mul` and is only provided
 * for convenience. 
 * */
template<typename Type>
NSL::Tensor<Type> mat_vec(const NSL::Tensor<Type> & matrix,  const NSL::Tensor<Type> & vector){
    return torch::matmul(matrix,vector);
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_MAT_VEC_HPP
