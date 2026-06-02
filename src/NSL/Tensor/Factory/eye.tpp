#ifndef NSL_TENSOR_FACTORY_EYE_TPP
#define NSL_TENSOR_FACTORY_EYE_TPP

#include "../tensor.hpp"

namespace NSL {

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<Type> eye(NSL::Device device, const NSL::size_t & N){
    // Use N without braces to avoid matching the IntArrayRef overload of torch::eye,
    // which would ignore the TensorOptions and place the tensor on CPU.
    torch::TensorOptions opt = device.device().template dtype<Type>();
    return torch::eye(N, opt);
}

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<Type> eye(const NSL::size_t & N){
    return torch::eye({N}, torch::TensorOptions().dtype<Type>());
}

} // namespace NSL

#endif //NSL_TENSOR_FACTORY_EYE_TPP
