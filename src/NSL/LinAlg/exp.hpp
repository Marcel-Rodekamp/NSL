#ifndef NSL_LINALG_EXP_HPP
#define NSL_LINALG_EXP_HPP

#include <cmath>
#include "../Tensor/tensor.hpp"

namespace NSL {
namespace LinAlg {

template<typename Type>
Type exp(Type number){
    return std::exp(number);
}

//exponential(Tensor)
template<typename Type>
NSL::Tensor<Type> exp(Tensor<Type> &tensor) {
    NSL::Tensor<Type> out(tensor);
    return out.exp();
}

// \todo The exponentiation of a TimeTensor may require more careful thought.
// But, for now, since TimeTensor = NSL::Tensor, this definition is redundant.
//exponential(Timetensor)
//template<typename Type>
//NSL::TimeTensor<Type> exp(TimeTensor<Type> &tensor) {
//    NSL::TimeTensor<Type> out(tensor);
//    return out.exp();
//}

} // namespace LinAlg
} // namespace NSL
#endif //NSL_LINALG_EXP_HPP
