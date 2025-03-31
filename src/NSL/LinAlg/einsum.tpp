#ifndef NSL_LINALG_EINSUM_HPP
#define NSL_LINALG_EINSUM_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg{

// template<NSL::Concept::isNumber Type>
// NSL::Tensor<Type> einsum(const std::string &equation, const std::vector<NSL::Tensor<Type>> &operands) {
//     return torch::einsum(equation, {operands});
// }

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> einsum(const std::string &equation, const NSL::Tensor<Type>& left, const NSL::Tensor<Type>& right) {
    return torch::einsum(equation, {left,right});
}
        
} // namespace NSL::LinAlg

#endif //NSL_EINSUM_HPP
