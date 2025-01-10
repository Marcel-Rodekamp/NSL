#ifndef NSL_LINALG_MAT_EXP_HPP
#define NSL_LINALG_MAT_EXP_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

template <typename Type> NSL::Tensor<Type> mat_exp(const NSL::Tensor<Type> & t){
     NSL::Tensor<Type> L, Q;
     std::tie(L,Q) = torch::linalg::eigh(t,"L");  // eigh returns a tuple L, Q, where L is the list of eigenvalues and Q the unitary transformation
                                                  // use the std::tie function to receive the tuple
     NSL::Tensor<double> Qr = Q.real();
     NSL::Tensor<Type> expL = torch::diag(L.exp());  // take exponential of eigenvalues and make matrix with these elements as diagonal
     expL = torch::matmul(expL,Qr.T());
     expL = torch::matmul(Qr,expL);   // expK = Q @ expL @Q.T.conj()
     return NSL::Tensor<Type>  (expL,true);
     //return NSL::Tensor<Type>(t,true).mat_exp();
}

} // namespace NSL::LinAlg

#endif
