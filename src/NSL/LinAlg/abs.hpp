#ifndef NSL_LATTICES_ABS_HPP
#define NSL_LATTICES_ABS_HPP


#include <cmath>
#include "../Tensor/tensor.hpp"
#include <torch/torch.h>

namespace NSL {
namespace LinAlg {

template<typename Type>
inline Type abs(Type number){
    return std::abs(number);
}

template<typename Type>
inline Type abs(NSL::complex<Type> number){
    return std::abs(number);
}

} // namespace LinAlg
} // namespace NSL
#endif //NSL_LATTICES_ABS_HPP
