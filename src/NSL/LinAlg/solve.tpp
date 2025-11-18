#ifndef NSL_LINALG_SOLVE_TPP
#define NSL_LINALG_SOLVE_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

template <NSL::Concept::isNumber Type>
NSL::Tensor<Type> solve_triangular(const NSL::Tensor<Type> & leftTensor, const NSL::Tensor<Type> & rightTensor, bool left = true){
    return torch::linalg_solve_triangular( leftTensor, rightTensor, true, left, false ); //linalg::solve_triangular( leftTensor, rightTensor, true, left, false ); 
}

template <NSL::Concept::isNumber Type>
NSL::Tensor<Type> solve(const NSL::Tensor<Type> & leftTensor, const NSL::Tensor<Type> & rightTensor, bool left = true){
    return torch::linalg_solve( leftTensor, rightTensor, left); //linalg::solve( leftTensor, rightTensor, left); 
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_SVD_TPP
