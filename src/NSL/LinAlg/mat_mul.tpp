#ifndef NSL_LINALG_MAT_MUL_HPP
#define NSL_LINALG_MAT_MUL_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg{

//! matrix multiplication
/*!
 * out = left @ right 
 * */
template<typename T>
NSL::Tensor<T> mat_mul(const NSL::Tensor<T> & left, const NSL::Tensor<T> & right){
    return torch::matmul(left,right);    
}
        
} // namespace NSL::LinAlg

#endif //NANOSYSTEMLIBRARY_MAT_MUL_HPP
