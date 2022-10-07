#ifndef NSL_LINALG_MAT_VEC_HPP
#define NSL_LINALG_MAT_VEC_HPP

#include "../Tensor.hpp"
#include "typePromotion.hpp"
#include <ATen/ops/matmul.h>

namespace NSL::LinAlg{

//! matrix times vector
/*!
 * Implementation is the same as `NSL::LinAlg::mat_mul` and is only provided
 * for convenience. 
 * */
template<NSL::Concept::isNumber MatrixType, NSL::Concept::isNumber VectorType>
NSL::Tensor<NSL::CommonTypeOf<MatrixType,VectorType>> mat_vec(
        const NSL::Tensor<MatrixType> & matrix,  
        const NSL::Tensor<VectorType> & vector){
    return torch::matmul(
        NSL::Tensor<NSL::CommonTypeOf<MatrixType,VectorType>>(matrix),
        NSL::Tensor<NSL::CommonTypeOf<MatrixType,VectorType>>(vector)
    );
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> mat_vec(const NSL::Tensor<Type> & matrix, const NSL::Tensor<Type> & vector){
    return torch::matmul(matrix,vector);
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_MAT_VEC_HPP
