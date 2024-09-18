#ifndef NSL_INNER_PRODUCT_TPP
#define NSL_INNER_PRODUCT_TPP

#include "../Tensor.hpp"
#include "conj.tpp"

namespace NSL::LinAlg{

//! Inner product of vector a and b.
/*! 
 * It is assumed that a,b are one dimensional. If not they are treated as if.
 * This function computes
 *
 * \f[
 *    <a,b> = \sum_{i=0}^{N} a_i^* \cdot b_i 
 * \f]
 *
 * */
template<NSL::Concept::isNumber LeftType, NSL::Concept::isNumber RightType>
NSL::CommonTypeOf<LeftType,RightType> inner_product(
        const NSL::Tensor<LeftType> & a,  
        const NSL::Tensor<RightType> & b){
    return (NSL::LinAlg::conj(
                NSL::Tensor<NSL::CommonTypeOf<LeftType,RightType>>(a)
            ) * 
                NSL::Tensor<NSL::CommonTypeOf<LeftType,RightType>>(b)
    ).sum();
}

template<NSL::Concept::isNumber Type>
Type inner_product(const NSL::Tensor<Type> & a, const NSL::Tensor<Type> & b ){
    return (NSL::LinAlg::conj(a)*b).sum();
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> inner_product(const NSL::Tensor<Type> & a, const NSL::Tensor<Type> & b , const NSL::size_t & dim){
    return (NSL::LinAlg::conj(a)*b).sum(dim);
}

//! Inner product of tensor a and b.
/*! 
 * Computes the dot product for 1D tensors. For higher dimensions, sums the product of elements from input and other along their last dimension.
 *
 * \f[
 *    <a,b> = \sum_{i=0}^{N} a_i^* \cdot b_i 
 * \f]
 *
 * */
template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> inner(const NSL::Tensor<Type> & a, const NSL::Tensor<Type> & b ){
    return torch::inner(NSL::LinAlg::conj(a),b);
}

} // namespace NSL::LinAlg


#endif //NSL_INNER_PRODUCT_TPP
