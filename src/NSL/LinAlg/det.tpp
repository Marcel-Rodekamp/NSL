#ifndef NSL_LINALG_DET_TPP
#define NSL_LINALG_DET_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

template <typename Type> 
Type det(const NSL::Tensor<Type> & t){
    //! \todo: if t is not a matrix we would have a stack of determinants: Handle this case.
    //! \todo: add det as a Tensor member
    return torch::det( t ).template item<Type>(); 
}

template <typename Type> 
Type logdet(const NSL::Tensor<Type> & t){
    //! \todo if t is not a matrix we would have a stack of determinants: Handle this case.
    //! \todo: add logdet as a Tensor member
    return torch::logdet(t).template item<Type>();
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_DET_TPP
