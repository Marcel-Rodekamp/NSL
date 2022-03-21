#ifndef NSL_LINALG_TRANSPOSE_HPP
#define NSL_LINALG_TRANSPOSE_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

template <typename Type> 
inline NSL::Tensor<Type> transpose(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(t,true).T();
}

template <typename Type> 
inline NSL::Tensor<Type> transpose(const NSL::Tensor<Type> & t, const size_t dim0, const size_t dim1){
    return NSL::Tensor<Type>(t,true).T(dim0, dim1);
}

} // namespace NSL::LinAlg

#endif
