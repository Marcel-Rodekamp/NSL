#ifndef NSL_LINALG_MINMAX_TPP
#define NSL_LINALG_MINMAX_TPP

namespace NSL::LinAlg{

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> min(const NSL::Tensor<Type> & t){
    return torch::min(t);
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> max(const NSL::Tensor<Type> & t){
    return torch::max(t);
}

}

#endif // NSL_LINALG_MINMAX_TPP
