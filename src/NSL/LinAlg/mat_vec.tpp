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

//! matrix times vector
/*!
 * Implementation is the same as `NSL::LinAlg::mat_mul` and is only provided
 * for convenience. 
 * \todo: Apply the typecasting discussed in PR #33 this should apply the PRECISE::type
 *        As the PR is not finished I leave a naive implementation always casting to complex
 * */
template<NSL::Concept::isNumber MatrixType, NSL::Concept::isNumber VectorType>
auto mat_vec(const NSL::Tensor<MatrixType> & matrix,  const NSL::Tensor<VectorType> & vector){
    if constexpr(NSL::is_complex<MatrixType> && !NSL::is_complex<VectorType> ){
        // Matrix is complex:
        return NSL::Tensor<MatrixType>(torch::matmul(matrix,vector));
    } else if constexpr(!NSL::is_complex<MatrixType> && NSL::is_complex<VectorType> ){
        // Vector is complex:
        return NSL::Tensor<VectorType>(torch::matmul(matrix,vector));
    } else{
        // both either real or complex
        return NSL::Tensor<MatrixType>(torch::matmul(matrix,vector));

    }
}


} // namespace NSL::LinAlg

#endif // NSL_LINALG_MAT_VEC_HPP
