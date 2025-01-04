#ifndef NSL_LINALG_DET_TPP
#define NSL_LINALG_DET_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

template <typename Type> 
Type det(const NSL::Tensor<Type> & t){
    //! \todo: if t is not a matrix we would have a stack of determinants: Handle this case.
    //! \todo: add det as a Tensor member
    return torch::det( t ).template item<Type>(); 
}

template <typename Type> 
Type logdet(const NSL::Tensor<Type> & t){
    //! \todo if t is not a matrix we would have a stack of determinants: Handle this case.
    //! \todo: add logdet as a Tensor member
    return torch::log(torch::det(t)).template item<Type>();
}

template <typename Type> 
Type logdet1plusF(const NSL::Tensor<Type> & t){
    //! \todo Need to mod the imaginary part back into the interval (-2pi,2pi)
    
    NSL::Tensor<Type> ev = torch::linalg::eigvals(t);  // eigenvalues of the sausage
    NSL::Tensor<Type> log1plusF=torch::log(1.0+ev);    // logarithm of 1+eigenvalues

    NSL::complex<double> logdet1pF = log1plusF.sum(); // sum the elements, which gives logdet
    logdet1pF = std::complex<double> (logdet1pF.real(), std::remainder(logdet1pF.imag(), 6.28318530717959)); // mod the imaginary part into -2pi,2pi

    return logdet1pF;  
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_DET_TPP
