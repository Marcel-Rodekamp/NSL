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

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> min_with_scalar(const NSL::Tensor<Type>& tensor, NSL::RealTypeOf<Type> value) {
    return torch::clamp(NSL::real(tensor), c10::nullopt, value).to(torch::kComplexDouble);
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> max_with_scalar(const NSL::Tensor<Type>& tensor, NSL::RealTypeOf<Type> value) {
    // return NSL::Tensor<Type> (torch::clamp(NSL::Tensor<NSL::RealTypeOf<Type>> (tensor), value, c10::nullopt));
    return torch::clamp(NSL::real(tensor), value, c10::nullopt).to(torch::kComplexDouble);
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_MINMAX_TPP
