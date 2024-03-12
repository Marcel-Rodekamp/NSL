#ifndef NSL_TENSOR_FACTORY_LIKE_TPP
#define NSL_TENSOR_FACTORY_LIKE_TPP

#include "../tensor.hpp"

namespace NSL {

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<Type> zeros_like(const NSL::Tensor<Type> & other){
    return torch::zeros_like(other);
}

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<Type> ones_like(const NSL::Tensor<Type> & other){
    return torch::ones_like(other);
}


} // namespace NSL

#endif //NSL_TENSOR_FACTORY_LIKE_TPP
