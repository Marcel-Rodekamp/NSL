#ifndef NSL_LINALG_LOG_HPP
#define NSL_LINALG_LOG_HPP

#include <cmath>
#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! log(Type)
template<NSL::Concept::isNumber Type>
Type log(Type number){
    return std::log(number);
}

//! log(Tensor) - element wise 
template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> log(const Tensor<Type> & t) {
    return NSL::Tensor<Type>(t,true).log();
}

} // namespace NSL::LinAlg

#endif //NSL_LINALG_LOG_HPP
