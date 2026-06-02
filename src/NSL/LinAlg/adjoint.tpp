#ifndef NSL_LINALG_ADJOINT_HPP
#define NSL_LINALG_ADJOINT_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

template <typename Type> 
inline NSL::Tensor<Type> adjoint(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(t,true).H();
}

template <typename Type> 
inline NSL::Tensor<Type> adjoint(const NSL::Tensor<Type> & t, const size_t dim0, const size_t dim1){
    return NSL::Tensor<Type>(t,true).H(dim0, dim1);
}

// Zero-copy conjugate-transpose view (no allocation).
// Safe to use when the result is consumed immediately (e.g. passed to solve/mat_mul)
// and the source tensor is not modified during the same expression.
template <typename Type>
inline NSL::Tensor<Type> adjoint_view(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(torch::adjoint(t));
}

template <typename Type>
inline NSL::Tensor<Type> adjoint_view(const NSL::Tensor<Type> & t, const size_t dim0, const size_t dim1){
    return NSL::Tensor<Type>(torch::transpose(t, dim0, dim1).conj());
}

} // namespace NSL::LinAlg

#endif
