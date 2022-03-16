#ifndef NSL_TENSOR_IMPL_PRINT_TPP
#define NSL_TENSOR_IMPL_PRINT_TPP

#include <iostream>

#include "base.tpp"

namespace NSL::TensorImpl {

//! Put the `NSL::Tensor` to the outstream
template<NSL::Concept::isNumber Type>
std::ostream & operator<<(std::ostream & os, const NSL::Tensor<Type> & tensor){
    //! \todo Add a better printer!
    os << tensor.data_;
    return os;
}

    

} // namespace NSL::TensorImpl

#endif //NSL_TENSOR_IMPL_PRINT_TPP
