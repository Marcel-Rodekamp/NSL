#ifndef NSL_LINALG_CEIL_HPP
#define NSL_LINALG_CEIL_HPP

#include <cmath>
#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! log(Type)
template<NSL::Concept::isNumber Type>
Type ceil(Type number){
    return std::ceil(number);
}

} // namespace NSL::LinAlg

#endif //NSL_LINALG_CEIL_HPP
