#include "fermionMatrixHubbardExp.hpp"


namespace NSL::FermionMatrix {

//!
template<typename Type>
NSL::TimeTensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::F_(const NSL::TimeTensor<Type> & psi){
    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1);

    //! \todo Add complex literals
    const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);

    // apply kronecker delta
    NSL::TimeTensor<Type> psiShift = NSL::LinAlg::shift(psi,1);

    NSL::TimeTensor<Type> Fpsi = NSL::LinAlg::mat_vec(
        //! \todo: This argument needs to be variable with the beta coming from elsewhere
        //!        CHANGE THAT!!!
        // exp_hopping_matrix computes only once and stores the result accessible with the same function
        this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */0.1),
        ((this->phi_*I).exp() * psiShift).transpose()
    );

    // anti-periodic boundary condition

    Fpsi.slice(0,0,1)*=-1;
    return Fpsi.transpose();
}

template <typename Type>
NSL::TimeTensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::M(const NSL::TimeTensor<Type> &psi){
    return psi - this->F_(psi);
}


//! \todo: Full support of complex number multiplication is missing in Tensor:
//template class NSL::FermionMatrix::FermionMatrixHubbardExp<float>;
//template class NSL::FermionMatrix::FermionMatrixHubbardExp<double>;
//template class NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<float>>;
template class NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<double>>;

} // namespace FermionMatrix
