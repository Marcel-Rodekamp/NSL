#ifndef NSL_LINALG_INV_TPP
#define NSL_LINALG_INV_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

template <typename Type> 
Type mat_inv(const NSL::Tensor<Type> & t){
    //! \todo: if t is not a matrix we would have a stack of determinants: Handle this case.
    //! \todo: add inv as a Tensor member
    return torch::inverse( t ).template item<Type>(); 
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_INV_TPP
