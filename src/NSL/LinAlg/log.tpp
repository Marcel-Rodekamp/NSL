#ifndef NSL_LINALG_LOG_HPP
#define NSL_LINALG_LOG_HPP

#include <cmath>
#include "../Tensor.hpp"

namespace NSL::LinAlg {

template<typename Type>
Type log(Type number){
    return std::log(number);
}

//logonential(Tensor)
template<typename Type>
NSL::Tensor<Type> log(const Tensor<Type> & t) {
    return NSL::Tensor<Type>(t,true).log();
}

} // namespace NSL::LinAlg

#endif //NSL_LINALG_LOG_HPP
