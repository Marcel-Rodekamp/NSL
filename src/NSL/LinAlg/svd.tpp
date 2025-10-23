#ifndef NSL_LINALG_SVD_TPP
#define NSL_LINALG_SVD_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! performs SVD decomposition
template <NSL::Concept::isNumber Type>
std::tuple< NSL::Tensor<Type>, NSL::Tensor<Type>, NSL::Tensor<Type> > svd(const NSL::Tensor<Type> & t){
    return torch::linalg_svd( t, "gesvd" );
//    return torch::linalg_svd( t, true, "gesvd" ); //linalg::svd( t, true, "gesvd" );
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_SVD_TPP
