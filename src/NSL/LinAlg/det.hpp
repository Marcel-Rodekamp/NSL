#ifndef NSL_LINALG_DET_HPP
#define NSL_LINALG_DET_HPP

#include "../Tensor/tensor.hpp"

namespace NSL {
namespace LinAlg {

template <typename Type> 
Type det(const NSL::Tensor<Type> & t){
    //! \todo if t is not a matrix we would have a stack of determinants: Handle this case.
    
    return to_torch(t).det().template item<Type>();
}

template <typename Type> 
Type logdet(const NSL::Tensor<Type> & t){
    //! \todo if t is not a matrix we would have a stack of determinants: Handle this case.

    return to_torch(t).det().log().template item<Type>();
}



} // namespace LinAlg
} // namespace NSL

#endif
