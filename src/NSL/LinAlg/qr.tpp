#ifndef NSL_LINALG_QR_TPP
#define NSL_LINALG_QR_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! returns eigenvalues and eigenvectors of symmetrix matrix M (assumes matrix is symmetric, and does NOT check for this)
//! eigenvalues and corresponding eigenvectors are sorted
template <NSL::Concept::isNumber Type>
std::tuple<NSL::Tensor<Type>,NSL::Tensor<Type>> qr(const NSL::Tensor<Type> & t){
    return torch::linalg::qr( t, "reduced" ); 
}

//! returns eigenvalues and eigenvectors of symmetrix matrix M (assumes matrix is symmetric, and does NOT check for this)
//! eigenvalues and corresponding eigenvectors are sorted
template <NSL::Concept::isNumber Type>
std::tuple<NSL::Tensor<Type>,NSL::Tensor<Type>,NSL::Tensor<Type>> udt(const NSL::Tensor<Type> & t){
    NSL::Tensor<Type> Q;
    NSL::Tensor<Type> R;
    std::tie( Q , R ) = NSL::LinAlg::qr( t );
    NSL::Tensor<Type> D = NSL::LinAlg::diag(R);
    NSL::Tensor<Type> V = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(1./D), R);
    return std::tie( Q, D, V ); 
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_QR_TPP
