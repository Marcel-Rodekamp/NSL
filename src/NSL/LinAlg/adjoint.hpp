#ifndef NSL_LINALG_ADJOINT_HPP
#define NSL_LINALG_ADJOINT_HPP

#include "../Tensor/tensor.hpp"

namespace NSL::LinAlg {

template <typename Type> 
inline NSL::Tensor<Type> adjoint(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(t).adjoint();
}

template <typename Type> 
inline NSL::Tensor<Type> adjoint(const NSL::Tensor<Type> & t, const size_t dim0, const size_t dim1){
    return NSL::Tensor<Type>(t).adjoint(dim0, dim1);
}

} // namespace NSL::LinAlg

#endif
