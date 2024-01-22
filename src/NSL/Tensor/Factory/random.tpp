#ifndef NSL_TENSOR_FACTORY_RANDOM_TPP
#define NSL_TENSOR_FACTORY_RANDOM_TPP

namespace NSL {

template<NSL::Concept::isNumber Type, typename ... TensorArgs>
NSL::Tensor<Type> rand(TensorArgs ... args){
    NSL::Tensor<Type> t(args...);
    
    t.rand();

    return t;
}

template<NSL::Concept::isNumber Type, typename ... tensorArgs>
NSL::Tensor<Type> randn(tensorArgs ... args){
    NSL::Tensor<Type> t(args...);
    
    t.randn();

    return t;
}

template<NSL::Concept::isNumber Type, typename ... TensorArgs>
NSL::Tensor<Type> randn(NSL::RealTypeOf<Type> mean, NSL::RealTypeOf<Type> std, TensorArgs ... args){
    NSL::Tensor<Type> t(args...);
    
    t.randn(mean,std);

    return t;
}

namespace Random {
template<NSL::Concept::isNumber Type>
Type rand(){
    NSL::Tensor<Type> t(1);
    
    t.rand();

    return t(0);
}

template<NSL::Concept::isNumber Type>
Type rand(NSL::RealTypeOf<Type> low, NSL::RealTypeOf<Type> high){
    NSL::Tensor<Type> t(1);
    
    t.rand(low,high);

    return t(0);
}

template<NSL::Concept::isNumber Type>
Type randn(){
    NSL::Tensor<Type> t(1);
    
    t.randn();

    return t(0);
}

template<NSL::Concept::isNumber Type>
Type randn(NSL::RealTypeOf<Type> mean, NSL::RealTypeOf<Type> std){
    NSL::Tensor<Type> t(1);
    
    t.randn(mean,std);

    return t();
}

} //namespace Random

} // namespace NSL

#endif // NSL_TENSOR_FACTORY_RANDOM_TPP
