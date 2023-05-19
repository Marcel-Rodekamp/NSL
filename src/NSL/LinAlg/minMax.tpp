#ifndef NSL_LINALG_MINMAX_TPP
#define NSL_LINALG_MINMAX_TPP

namespace NSL::LinAlg{

//! Computes the minimum of all elements of the tensor.
/*!
 * This function compute the minimum of all elements of the Tensor t. 
 * If Type is complex a runtime error is thrown. 
 * */
template<NSL::Concept::isNumber Type>
Type min(const NSL::Tensor<Type> & t){
    return torch::min(t).template item<Type>();
}

//! Computes the minimum of all elements of the tensor.
/*!
 * This function compute the minimum of all elements of the Tensor t. 
 * If Type is complex a runtime error is thrown. 
 * */
template<NSL::Concept::isNumber Type>
Type max(const NSL::Tensor<Type> & t){
    return torch::max(t).template item<Type>();
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_MINMAX_TPP
