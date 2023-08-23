#ifndef NSL_LINALG_EXP_HPP
#define NSL_LINALG_EXP_HPP

#include <cmath>
#include "../Tensor.hpp"
#include "Configuration.hpp"

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

template<NSL::Concept::isNumber Type>
NSL::Configuration<Type> exp( const NSL::Configuration<Type> & t_) {
    NSL::Configuration<Type> t(t_,true);

    for(auto &[key,field]: t){
        field = field.exp();
    } 

    return t;
}

} // namespace NSL::LinAlg

#endif //NSL_LINALG_EXP_HPP
