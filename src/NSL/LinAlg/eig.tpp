#ifndef NSL_LINALG_EIG_TPP
#define NSL_LINALG_EIG_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! returns eigenvalues and eigenvectors of symmetrix matrix M (assumes matrix is symmetric, and does NOT check for this)
//! eigenvalues and corresponding eigenvectors are sorted
template <NSL::Concept::isNumber Type>
std::tuple<NSL::Tensor<Type>,NSL::Tensor<Type>> eig(const NSL::Tensor<Type> & t){
    //! \todo: add eigh as a Tensor member
    return torch::linalg_eig( t ); 
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_EIG_TPP
