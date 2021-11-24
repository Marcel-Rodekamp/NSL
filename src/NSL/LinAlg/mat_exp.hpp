#ifndef NSL_LINALG_MAT_EXP_HPP
#define NSL_LINALG_MAT_EXP_HPP

#include "../Tensor/tensor.hpp"

namespace NSL {
namespace LinAlg {

template <typename Type> NSL::TimeTensor<Type> mat_exp(const NSL::TimeTensor<Type> & t){
    NSL::TimeTensor<Type> out(t);
    return out.mat_exp();
}


template <typename Type> NSL::TimeTensor<Type> exp(const NSL::TimeTensor<Type> & t){
    NSL::TimeTensor<Type> out(t);
    return out.exp();
}

} // namespace LinAlg
} // namespace NSL

#endif
