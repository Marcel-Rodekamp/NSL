#ifndef NSL_LINALG_ADJOINT_HPP
#define NSL_LINALG_ADJOINT_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

template <typename Type> 
inline NSL::Tensor<Type> adjoint(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(t,true).H();
}

template <typename Type> 
inline NSL::Tensor<Type> adjoint(const NSL::Tensor<Type> & t, const size_t dim0, const size_t dim1){
    return NSL::Tensor<Type>(t,true).H(dim0, dim1);
}

} // namespace NSL::LinAlg

#endif
