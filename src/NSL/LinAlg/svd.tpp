#ifndef NSL_LINALG_SVD_TPP
#define NSL_LINALG_SVD_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! returns eigenvalues and eigenvectors of symmetrix matrix M (assumes matrix is symmetric, and does NOT check for this)
//! eigenvalues and corresponding eigenvectors are sorted
template <NSL::Concept::isNumber Type>
std::tuple<NSL::Tensor<Type>,NSL::Tensor<Type>,NSL::Tensor<Type>> svd(const NSL::Tensor<Type> & t){
    return torch::linalg::svd( t, true, "gesvd" ); 
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_SVD_TPP
