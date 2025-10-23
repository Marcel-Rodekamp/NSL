#ifndef NSL_TENSOR_FACTORY_ZEROS_TPP
#define NSL_TENSOR_FACTORY_ZEROS_TPP

#include "../tensor.hpp"

namespace NSL {

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<Type> zeros(NSL::Device device, const NSL::size_t & N1, const NSL::size_t & N2){
    return torch::zeros({N1,N2},device.device().dtype<Type>() );
}

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<Type> zeros(const NSL::size_t & N1, const NSL::size_t & N2){
    return torch::zeros({N1,N2}, torch::TensorOptions().dtype<Type>());
}

} // namespace NSL

#endif //NSL_TENSOR_FACTORY_ZEROS_TPP
