#ifndef NSL_LINALG_MAT_EXP_HPP
#define NSL_LINALG_MAT_EXP_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

template <typename Type> NSL::Tensor<Type> mat_exp(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(t,true).mat_exp();
}

} // namespace NSL::LinAlg

#endif
