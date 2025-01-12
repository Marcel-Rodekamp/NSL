#ifndef NSL_LINALG_MAT_EXP_HPP
#define NSL_LINALG_MAT_EXP_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

template <typename Type> NSL::Tensor<Type> mat_exp(const NSL::Tensor<Type> & t){
     torch::Tensor L, Q;
     std::tie(L,Q) = torch::linalg::eigh(t,"L");  // eigh returns a tuple L, Q, where L is the list of eigenvalues and Q the unitary transformation
                                                  // use the std::tie function to receive the tuple
     L = torch::complex(L,torch::zeros_like(L));  // torch returns L as a tensor of doubles, but we need the tensor to be complex
                                                  // (there is probably a better way to make the array complex)
     torch::Tensor expL = torch::diag(L.exp());   // take exponential of eigenvalues and make matrix with these elements as diagonal
     expL = torch::matmul(expL,torch::transpose(Q,0,1).conj());
     expL = torch::matmul(Q,expL);                // expK = Q @ expL @ Q.T.conj()
     return NSL::Tensor<Type>  (expL,true);
     //return NSL::Tensor<Type>(t,true).mat_exp(); // this was the old way of doing it. . .
}

} // namespace NSL::LinAlg

#endif
