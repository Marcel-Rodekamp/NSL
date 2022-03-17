#ifndef NSL_LINALG_EXP_HPP
#define NSL_LINALG_EXP_HPP

#include <cmath>
#include "../Tensor.hpp"

namespace NSL::LinAlg {

template<typename Type>
Type exp(Type number){
    return std::exp(number);
}

//exponential(Tensor)
template<typename Type>
NSL::Tensor<Type> exp(const Tensor<Type> & t) {
    return NSL::Tensor<Type>(t,true).exp();
}

} // namespace NSL::LinAlg

#endif //NSL_LINALG_EXP_HPP
