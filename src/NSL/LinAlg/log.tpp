#ifndef NSL_LINALG_LOG_HPP
#define NSL_LINALG_LOG_HPP

#include <cmath>
#include "../Tensor.hpp"
#include "Configuration.hpp"

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

template<NSL::Concept::isNumber Type>
NSL::Configuration<Type> log( const NSL::Configuration<Type> & t_) {
    NSL::Configuration<Type> t(t_,true);

    for(auto &[key,field]: t){
        field.log();
    } 

    return t;
}

} // namespace NSL::LinAlg

#endif //NSL_LINALG_LOG_HPP
