#ifndef NSL_LINALG_QR_TPP
#define NSL_LINALG_QR_TPP

#include "../Tensor.hpp"

namespace NSL::LinAlg {

//! returns QR decomposition of matrix M (see https://docs.pytorch.org/cppdocs/api/function_namespaceat_1a4bcb45636b68191bccfe624ba0e97166.html  (though this won't tell you much!) )
template <NSL::Concept::isNumber Type>
std::tuple<NSL::Tensor<Type>,NSL::Tensor<Type>> qr(const NSL::Tensor<Type> & t){
    return torch::linalg_qr( t, "reduced" ); //linalg::qr( t, "reduced" ); 
}

//! returns QDV decomposition of matrix M, where D is diagonal (R = DV)
template <NSL::Concept::isNumber Type>
std::tuple<NSL::Tensor<Type>,NSL::Tensor<Type>,NSL::Tensor<Type>> udt(const NSL::Tensor<Type> & t){
    NSL::Tensor<Type> Q = NSL::zeros_like(t);
    NSL::Tensor<Type> R = NSL::zeros_like(t);
    std::tie( Q , R ) = NSL::LinAlg::qr( t );
    NSL::Tensor<Type> D = NSL::LinAlg::diagonal(R);
    // NSL::Tensor<Type> V = NSL::LinAlg::mat_mul(NSL::LinAlg::diag_embed(1./D), R);
    R *= (1./D).expand_view(D.shape(D.dim()-1), R.dim()-1);
    return std::tie( Q, D, R );
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_QR_TPP
