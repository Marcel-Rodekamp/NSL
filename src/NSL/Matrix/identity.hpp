#ifndef NSL_MATRIX_IDENTITY_HPP
#define NSL_MATRIX_IDENTITY_HPP

#include "../Tensor.hpp"
#include <ATen/core/ATen_fwd.h>
#include <torch/torch.h>

namespace NSL::Matrix {

template <typename Type>
inline NSL::Tensor<Type> Identity(const NSL::size_t & size){
    return NSL::Tensor<Type>(torch::eye(
                size, 
                torch::TensorOptions().dtype<Type>()
                ));
}

template <typename Type>
inline NSL::Tensor<Type> Identity(const NSL::Device & dev,const NSL::size_t & size){
    torch::TensorOptions opt = dev.device();

    return NSL::Tensor<Type>(torch::eye(
                size, 
                opt.dtype<Type>()
    ));
}

}

#endif
