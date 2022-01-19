#ifndef NSL_LINALG_CONJ_HPP
#define NSL_LINALG_CONJ_HPP

#include <cmath>
#include "../Tensor/tensor.hpp"

namespace NSL {
namespace LinAlg {

template<typename Type>
NSL::TimeTensor<Type>  conj(const NSL::TimeTensor<Type> & tensor) {
    NSL::TimeTensor<Type> out(tensor);
    return out.conj();
}

template<typename Type>
NSL::TimeTensor<Type>  adjoint(const NSL::TimeTensor<Type> & tensor) {
    NSL::TimeTensor<Type> out(tensor);
    return(out.adjoint());
}

}}
#endif