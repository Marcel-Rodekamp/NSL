#ifndef NSL_LINALG_MAT_EXP_HPP
#define NSL_LINALG_MAT_EXP_HPP

#include "../Tensor/tensor.hpp"

namespace NSL {
namespace LinAlg {

template <typename Type> NSL::Tensor<Type> mat_exp(const NSL::Tensor<Type> & t){
    NSL::Tensor<Type> out(t);
    return out.mat_exp();
}

} // namespace LinAlg
} // namespace NSL

#endif
