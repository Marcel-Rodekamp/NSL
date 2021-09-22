#ifndef NSL_LINALG_DET_HPP
#define NSL_LINALG_DET_HPP

#include "../Tensor/tensor.hpp"

namespace NSL {
namespace LinAlg {

template <typename Type> NSL::Tensor<Type> det(const NSL::Tensor<Type> & t){
    NSL::Tensor<Type> out(t);
    return out.det();
}

} // namespace LinAlg
} // namespace NSL

#endif
