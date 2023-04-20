#ifndef NSL_LINALG_INV_TPP
#define NSL_LINALG_INV_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! matrix^-1
//template<NSL::Concept::isNumber MatrixType>
//NSL::Tensor<NSL::CommonTypeOf<MatrixType>> mat_inv(
//        const NSL::Tensor<MatrixType> & M ){
//    return torch::inverse(
//        NSL::Tensor<NSL::CommonTypeOf<MatrixType>>(M)
//    );
//}


template <NSL::Concept::isNumber Type>
NSL::Tensor<Type> mat_inv(const NSL::Tensor<Type> & t){
    //! \todo: add inv as a Tensor member
    return torch::inverse( t ); 
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_INV_TPP
