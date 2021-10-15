#include "Tensor/tensor.hpp"


// The explicit template instantiation forces the compiler to compile everything once
// thus even function which we don't test explicitly.

template class NSL::Tensor<bool>;
template class NSL::Tensor<int>;
template class NSL::Tensor<float>;
template class NSL::Tensor<double>;
template class NSL::Tensor<NSL::complex<float>>;
template class NSL::Tensor<NSL::complex<double>>;
// Other datatypes are not permitted by torch