#ifndef NSL_LINALG_MINMAX_TPP
#define NSL_LINALG_MINMAX_TPP

namespace NSL::LinAlg{

//! Computes the minimum of all elements of the tensor.
/*!
 * This function compute the minimum of all elements of the Tensor t. 
 * If Type is complex a runtime error is thrown. 
 * */
template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> clamp(const NSL::Tensor<Type>& tensor, NSL:RealTypeOf<Type> min_value, NSL:RealTypeOf<Type> max_value) {
    return torch::clamp(NSL::Tensor<NSL:RealTypeOf<Type>>(tensor), min_value, max_value);
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_MINMAX_TPP
