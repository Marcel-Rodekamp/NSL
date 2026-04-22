#ifndef NSL_LINALG_DIAG_TPP
#define NSL_LINALG_DIAG_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! return the diagonal elements of a NSL::Tensor with square dimension
template <typename Type> 
inline NSL::Tensor<Type> diag(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(torch::diag(t),true);
}

//! return the diagonal elements of a NSL::Tensor with batched square dimension
template <typename Type> 
inline NSL::Tensor<Type> diagonal(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(torch::diagonal(t, 0, -2,-1),true);
}

//! return the diagonal elements of a NSL::Tensor with square dimension
template <typename Type>
inline NSL::Tensor<Type> diag_embed(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(torch::diag_embed(t, 0, -2,-1),true);
}

//! Add values d to the main diagonal of the last two dimensions of t, in-place.
//! Avoids allocating a full diag_embed(d) matrix (~Nt*Nx*Nx elements) for the addition.
//! Equivalent to: t += diag_embed(d), but zero extra allocation.
//! Accepts both lvalues and rvalues (e.g. slice expressions) via pass-by-value;
//! the shallow copy shares underlying storage so the modification is visible in the caller.
template <typename Type>
inline NSL::Tensor<Type> add_diagonal(NSL::Tensor<Type> t, const NSL::Tensor<Type> & d){
    torch::diagonal(t, 0, -2, -1) += d;
    return t;
}

} // namespace NSL::LinAlg

#endif
