#ifndef NSL_LINALG_CONJ_HPP
#define NSL_LINALG_CONJ_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

template <typename Type> 
inline NSL::Tensor<Type> conj(const NSL::Tensor<Type> & t){
    return NSL::Tensor<Type>(t,true).conj();
}

} // namespace NSL::LinAlg

#endif
