#ifndef NSL_TENSOR_FACTORY_EYE_TPP
#define NSL_TENSOR_FACTORY_EYE_TPP

#include "../tensor.hpp"

namespace NSL {

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<Type> eye(NSL::Device device, const NSL::size_t & N){
    return torch::eye({N},device.device().dtype<Type>() );
}

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<Type> eye(const NSL::size_t & N){
    return torch::eye({N}, torch::TensorOptions().dtype<Type>());
}

} // namespace NSL

#endif //NSL_TENSOR_FACTORY_EYE_TPP
