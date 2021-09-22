#ifndef NSL_LINALG_DET_HPP
#define NSL_LINALG_DET_HPP

#include "../Tensor/tensor.hpp"

namespace NSL {
namespace LinAlg {

template <typename Type> const NSL::Tensor<Type> det(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(to_torch(t).det());
}

template <typename Type> const NSL::Tensor<Type> logdet(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(to_torch(t).logdet());
}

} // namespace LinAlg
} // namespace NSL

#endif
