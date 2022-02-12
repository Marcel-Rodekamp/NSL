#ifndef NSL_LINALG_MATRIX_HPP
#define NSL_LINALG__MATRIX_HPP

#include "../Tensor/tensor.hpp"

namespace NSL {
namespace LinAlg {
namespace Matrix{

template <typename Type> 
    NSL::TimeTensor<Type> Identity(NSL::TimeTensor<Type> & t, const size_t & size ) {
        NSL::TimeTensor<Type> out(torch::eye(size, torch::TensorOptions().dtype<Type>()));
        return (out);
    }


} //namespace Matrix
} // namespace LinAlg
} // namespace NSL

#endif
