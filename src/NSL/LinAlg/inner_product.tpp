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
template<NSL::Concept::isNumber Type>
Type inner_product(const NSL::Tensor<Type> & a, const NSL::Tensor<Type> & b ){
    return (NSL::LinAlg::conj(a)*b).sum();

}

} // namespace NSL::LinAlg


#endif //NSL_INNER_PRODUCT_TPP
