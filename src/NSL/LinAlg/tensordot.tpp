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

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<Type> tensordot( const NSL::Tensor<Type> & left, const NSL::Tensor<Type> & right, NSL::size_t dim){

    return torch::tensordot(
        left,right,{dim},{dim}
    );

}

template<NSL::Concept::isNumber Type>
inline NSL::Tensor<Type> tensordot( const NSL::Tensor<Type> & left, const NSL::Tensor<Type> & right, NSL::size_t dimLeft, NSL::size_t dimRight){

    return torch::tensordot(
        left,right,{dimLeft},{dimRight}
    );

}

} //namespace NSL::LinAlg

#endif // NSL_LINALG_TENSORDOT_TPP
