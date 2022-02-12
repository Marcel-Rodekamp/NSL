#include "fermionMatrixHubbardExp.hpp"

namespace NSL::FermionMatrix {

//!
template<typename Type>
NSL::TimeTensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::F_(const NSL::TimeTensor<Type> & psi){
    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1);

    //! \todo Add complex literals
    //const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);

    // apply kronecker delta
    NSL::TimeTensor<Type> psiShift = NSL::LinAlg::shift(psi,1);

    NSL::TimeTensor<Type> Fpsi = NSL::LinAlg::mat_vec(
        //! \todo: This argument needs to be variable with the beta coming from elsewhere
        //!        CHANGE THAT!!!
        // exp_hopping_matrix computes only once and stores the result accessible with the same function
        this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */delta_),
        (this->phiExp_ * psiShift).transpose()
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
    //const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);

    // apply kronecker delta
    NSL::TimeTensor<Type> psiShift = NSL::LinAlg::shift(psi,1);

    out= NSL::LinAlg::mat_vec(
        NSL::LinAlg::adjoint(this->phiExp_).transpose(),
         this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */delta_))
     * psiShift;
    //anti-periodic boundary condition
    out.slice(0,0,1)*=-1;
    return (psi-out);
}

template<typename Type>
NSL::TimeTensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::MMdagger(const NSL::TimeTensor<Type> & psi){
    NSL::TimeTensor<Type> out;
    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1);
    
   //MMdagger(psi) = M(psi) + Mdagger(psi) + exp_hopping_mtrix^2 x psi - psi
    out= this->M(psi) + this->Mdagger(psi) + NSL::LinAlg::mat_vec((this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */delta_))*
        (this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */delta_)), NSL::LinAlg::mat_transpose(psi)).
    transpose();

    return (out-psi);

}

template<typename Type>
NSL::TimeTensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::MdaggerM(const NSL::TimeTensor<Type> & psi){
    NSL::TimeTensor<Type> out;
    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1);
    //const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);

    //MMdagger(psi) = M(psi) + Mdagger(psi) + exp(-i*phi) * (exp_hopping_mtrix^2 x exp(-i*phi)* psi) - psi
    out= this->M(psi) + this->Mdagger(psi) + (NSL::LinAlg::adjoint(this->phiExp_) *
        NSL::LinAlg::mat_vec((this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */delta_))*
         (this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */delta_)),
        ((this->phiExp_ * psi).transpose()))).transpose();

    return (out-psi);

}

//return type
template<typename Type>
Type NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::logDetM(){
    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1); 

    //For identity matrix 
    NSL::TimeTensor<Type> Id(Nx,Nx);
    const Type length = Nx;
    
    NSL::TimeTensor<Type> prod(Nt,Nx,Nx);
    NSL::TimeTensor<Type> out(Nx,Nx);
    Type logdet = 0.0;
    
    prod = this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */this->delta_)* NSL::LinAlg::shift(this->phiExp_,-1).expand(Nx).transpose(1,2);
    //F_{Nt-1}
    out = prod.slice(/*dim=*/0,/*start=*/Nt-1,/*end=*/Nt);

    //Computing F_{Nt-1}.F_{Nt-2}.....F_0
    for(int t=Nt-2; t>=0; t--){

        //! \todo: figure out what mat_mul does
        //out = NSL::LinAlg::mat_vec(out,prod.slice(/*dim=*/0,/*start=*/t,/*end=*/t+1));
        out.mat_mul(prod.slice(/*dim=*/0,/*start=*/t,/*end=*/t+1)); 
         
    }
      
    //out += Id.Identity(Nx);
    out += NSL::LinAlg::Matrix::Identity(Id, Nx);   
    logdet = NSL::LinAlg::logdet(out);

    return logdet;

}



//! \todo: Full support of complex number multiplication is missing in Tensor:
//template class NSL::FermionMatrix::FermionMatrixHubbardExp<float>;
//template class NSL::FermionMatrix::FermionMatrixHubbardExp<double>;
template class NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<float>>;
template class NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<double>>;

} // namespace FermionMatrix
