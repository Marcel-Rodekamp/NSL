#ifndef NSL_LINALG_SVD_TPP
#define NSL_LINALG_SVD_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! performs SVD decomposition
template <NSL::Concept::isNumber Type>
std::tuple< NSL::Tensor<Type>, NSL::Tensor<Type>, NSL::Tensor<Type> > svd(const NSL::Tensor<Type> & t){
    return torch::linalg_svd( t, "gesvd" );
    // this original call with the argument "true" does not work anymore T.L.  (remind me again why we use libtorch???)
    // return torch::linalg_svd( t, true, "gesvd" );
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_SVD_TPP
