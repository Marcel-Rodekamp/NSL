#ifndef NSL_LINALG_MAT_MUL_HPP
#define NSL_LINALG_MAT_MUL_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg{


//! matrix @ matrix / tensor @ tensor
/*!
 * Implementation is the same as `NSL::LinAlg::mat_vec` 
 * */
template<NSL::Concept::isNumber MatrixType, NSL::Concept::isNumber VectorType>
NSL::Tensor<NSL::CommonTypeOf<MatrixType,VectorType>> mat_mul(
        const NSL::Tensor<MatrixType> & leftTensor,  
        const NSL::Tensor<VectorType> & rightTensor){
    return torch::matmul(
        NSL::Tensor<NSL::CommonTypeOf<MatrixType,VectorType>>(leftTensor),
        NSL::Tensor<NSL::CommonTypeOf<MatrixType,VectorType>>(rightTensor)
    );
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> mat_mul(const NSL::Tensor<Type> & leftTensor, const NSL::Tensor<Type> & rightTensor){
    return torch::matmul(leftTensor,rightTensor);
}
        
} // namespace NSL::LinAlg

#endif //NANOSYSTEMLIBRARY_MAT_MUL_HPP
