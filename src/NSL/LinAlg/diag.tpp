#ifndef NSL_LINALG_DIAG_TPP
#define NSL_LINALG_DIAG_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! return the diagonal elements of a NSL::Tensor with square dimension
template <typename Type> 
inline NSL::Tensor<Type> diag(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(torch::diag(t),true);
}

} // namespace NSL::LinAlg

#endif
