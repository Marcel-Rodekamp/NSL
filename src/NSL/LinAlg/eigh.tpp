#ifndef NSL_LINALG_EIGH_TPP
#define NSL_LINALG_EIGH_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! returns eigenvalues and eigenvectors of symmetrix matrix M (assumes matrix is symmetric, and does NOT check for this)
//! eigenvalues and corresponding eigenvectors are sorted
template <NSL::Concept::isNumber Type>
std::tuple<NSL::Tensor<Type>,NSL::Tensor<Type>> eigh(const NSL::Tensor<Type> & t){
    //! \todo: add eigh as a Tensor member
    return torch::linalg::eigh( t, "L" ); 
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_EIGH_TPP
