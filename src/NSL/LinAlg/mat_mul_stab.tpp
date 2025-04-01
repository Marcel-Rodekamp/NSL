#ifndef NSL_LINALG_MAT_MUL_STAB_HPP
#define NSL_LINALG_MAT_MUL_STAB_HPP

#include "../Tensor.hpp"

namespace NSL::LinAlg{

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> mat_mul_stab(const NSL::Tensor<Type> & leftTensor, const NSL::Tensor<Type> & rightTensor){
    NSL::Tensor<Type> Q = NSL::zeros_like(leftTensor);
    NSL::Tensor<Type> D( leftTensor.device(), leftTensor.shape(0) );
    NSL::Tensor<Type> V = NSL::zeros_like(leftTensor);
    std::tie( Q, D, V ) = NSL::LinAlg::udt( leftTensor );

    NSL::Tensor<Type> Qprime = NSL::zeros_like(rightTensor);
    NSL::Tensor<Type> Dprime( rightTensor.device(), rightTensor.shape(0) );
    NSL::Tensor<Type> Vprime = NSL::zeros_like(rightTensor);
    std::tie( Qprime, Dprime, Vprime ) = NSL::LinAlg::udt( rightTensor );

    NSL::Tensor<Type> Qnew = NSL::zeros_like(rightTensor);
    NSL::Tensor<Type> Dnew( rightTensor.device(), rightTensor.shape(0) );
    NSL::Tensor<Type> Vnew = NSL::zeros_like(rightTensor);
    std::tie( Qnew, Dnew, Vnew ) = NSL::LinAlg::udt(  NSL::LinAlg::mat_mul( NSL::LinAlg::diag(D),NSL::LinAlg::mat_mul( NSL::LinAlg::mat_mul( V,Qprime ),NSL::LinAlg::diag(Dprime) ) )  );

    return NSL::LinAlg::mat_mul( 
        NSL::LinAlg::mat_mul( Q,Qnew ),NSL::LinAlg::mat_mul( NSL::LinAlg::diag(Dnew),NSL::LinAlg::mat_mul( Vnew,Vprime ) ) 
    );
}
        
} // namespace NSL::LinAlg

#endif //NANOSYSTEMLIBRARY_MAT_MUL_STAB_HPP
