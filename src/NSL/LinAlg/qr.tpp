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
    // Avoid pre-allocating zeros_like(t) — qr() already allocates its own output tensors.
    auto [Q, R] = NSL::LinAlg::qr( t );
    NSL::Tensor<Type> D = NSL::LinAlg::diagonal(R);
    // R <- diag(1/D) R, the V of the QDV decomposition, without forming diag(1/D). The 1 has the precision of Type:
    // a double 1. makes 1/D a complex<double> tensor for Type = complex<float>, and R *= of a wider type leaves R
    // unchanged (Tensor's compound operators compute in a converted copy then)
    R *= (NSL::RealTypeOf<Type>(1)/D).expand_view(D.shape(D.dim()-1), R.dim()-1);
    return std::make_tuple( std::move(Q), std::move(D), std::move(R) );
}

} // namespace NSL::LinAlg

#endif // NSL_LINALG_QR_TPP
