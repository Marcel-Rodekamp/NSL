#ifndef NSL_LINALG_SQRT_HPP
#define NSL_LINALG_SQRT_HPP

#include <cmath>
#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! log(Type)
template<NSL::Concept::isNumber Type>
Type sqrt(Type number){
    return std::sqrt(number);
}

//! log(Tensor) - element wise 
template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> sqrt(const Tensor<Type> & t) {
    return NSL::Tensor<Type>(t,true).sqrt();
}

} // namespace NSL::LinAlg

#endif //NSL_LINALG_SQRT_HPP