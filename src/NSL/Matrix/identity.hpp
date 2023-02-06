#ifndef NSL_MATRIX_IDENTITY_HPP
#define NSL_MATRIX_IDENTITY_HPP

#include "../Tensor.hpp"
#include <torch/torch.h>

namespace NSL::Matrix {

template <typename Type>
inline NSL::Tensor<Type> Identity(const NSL::size_t & size){
    return NSL::Tensor<Type>(torch::eye(
                size, 
                torch::TensorOptions().dtype<Type>()
                ));
}

}

#endif
