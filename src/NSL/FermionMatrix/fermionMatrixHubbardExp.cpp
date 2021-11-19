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
    ).transpose();

    // anti-periodic boundary condition

    Fpsi.slice(0,0,1)*=-1;
    return Fpsi;
}

template <typename Type>
NSL::TimeTensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::M(const NSL::TimeTensor<Type> & psi){
    return psi - this->F_(psi);
}

template<typename Type>
NSL::TimeTensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::Mdagger(const NSL::TimeTensor<Type> & psi){
    NSL::TimeTensor<Type> out;
    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1);
    const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);

    NSL::TimeTensor<Type> psiShift = NSL::LinAlg::shift(psi,1);
    out= NSL::LinAlg::mat_vec(NSL::LinAlg::adjoint((this->phi_*I).exp()).transpose(), this->Lat->exp_hopping_matrix(0.1)) * psiShift;
    //anti-periodic boundary condition
    out.slice(0,0,1)*=-1;
    return (psi-out);
}

template<typename Type>
NSL::TimeTensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::MMdagger(const NSL::TimeTensor<Type> & psi){
    NSL::TimeTensor<Type> out, outdagger;
    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1);
    const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);
    NSL::TimeTensor<Type> psiShift = NSL::LinAlg::shift(psi,1);

    //Since, phi.exp() changes phi_, copy it in phi because there are two exponential terms
    //Or should mat_exp() be called?? mat_exp does exp( matrix A) instead of elemetwise exponentiation
    NSL::TimeTensor<Type> phi = this->phi_;

    out= (NSL::LinAlg::mat_vec(this->Lat->exp_hopping_matrix(0.1),((phi*I).exp()).transpose()));
    out.transpose();
    //anti-periodic boundary condition
    out.slice(0,0,1)*=-1;

    outdagger= NSL::LinAlg::adjoint((this->phi_*I).exp()).transpose();
    outdagger=  NSL::LinAlg::mat_vec(outdagger, this->Lat->exp_hopping_matrix(0.1)) * psiShift;
    //anti-periodic boundary condition
    outdagger.slice(0,0,1)*=-1;

    out= out*outdagger;
    return (psi+out);
}

template<typename Type>
NSL::TimeTensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::MdaggerM(const NSL::TimeTensor<Type> & psi){
    NSL::TimeTensor<Type> out, outdagger;
    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1);
    const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);
    NSL::TimeTensor<Type> psiShift = NSL::LinAlg::shift(psi,1);

    //Since, phi.exp() changes phi_, copy it in phi because there are two exponential terms
    NSL::TimeTensor<Type> phi = this->phi_;

    outdagger=  NSL::LinAlg::mat_vec(NSL::LinAlg::adjoint((phi*I).exp()).transpose(), this->Lat->exp_hopping_matrix(0.1));
    //anti-periodic boundary condition
    outdagger.slice(0,0,1)*=-1;

    out=  NSL::LinAlg::mat_vec(this->Lat->exp_hopping_matrix(0.1), ((this->phi_*I).exp()).transpose());
    out= out.transpose() * psiShift;
    //anti-periodic boundary condition
    out.slice(0,0,1)*=-1;

    outdagger = outdagger * out;
    return (psi+outdagger);
}

//! \todo: Full support of complex number multiplication is missing in Tensor:
//template class NSL::FermionMatrix::FermionMatrixHubbardExp<float>;
//template class NSL::FermionMatrix::FermionMatrixHubbardExp<double>;
template class NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<float>>;
template class NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<double>>;

} // namespace FermionMatrix
