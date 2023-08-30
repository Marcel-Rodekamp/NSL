#ifndef NSL_LINALG_TENSORDOT_TPP
#define NSL_LINALG_TENSORDOT_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg{

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<Type> tensordot( const NSL::Tensor<Type> & left, const NSL::Tensor<Type> & right, std::vector<NSL::size_t> dimsLeft, std::vector<NSL::size_t> dimsRight){

    return torch::tensordot(
        left,right,dimsLeft,dimsRight
    );

}

} //namespace NSL::LinAlg

#endif // NSL_LINALG_TENSORDOT_TPP
