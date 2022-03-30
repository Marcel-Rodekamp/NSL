#ifndef NSL_FERMION_MATRIX_HUBBARD_EXP_TPP
#define NSL_FERMION_MATRIX_HUBBARD_EXP_TPP

#include "hubbardExp.hpp"
#include "../../Matrix.hpp"

namespace NSL::FermionMatrix {

//!
template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::F_(const NSL::Tensor<Type> & psi){
    const NSL::size_t Nt = this->phi_.shape(0);
    const NSL::size_t Nx = this->phi_.shape(1);

    //! \todo Add complex literals
    //const NSL::complex<NSL::Concept::isNumber RT_extractor<Type>::value_type> I(0,1);

    // apply kronecker delta
    NSL::Tensor<Type> psiShift = NSL::LinAlg::shift(psi,1);

    NSL::Tensor<Type> Fpsi = NSL::LinAlg::mat_vec(
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

template <NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::M(const NSL::Tensor<Type> & psi){
    return psi - this->F_(psi);
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::Mdagger(const NSL::Tensor<Type> & psi){
    NSL::Tensor<Type> out;
    const NSL::size_t Nt = this->phi_.shape(0);
    const NSL::size_t Nx = this->phi_.shape(1);
    //const NSL::complex<NSL::Concept::isNumber RT_extractor<Type>::value_type> I(0,1);

    // apply kronecker delta
    NSL::Tensor<Type> psiShift = NSL::LinAlg::shift(psi,1);

    out= NSL::LinAlg::mat_vec(
        NSL::LinAlg::adjoint(this->phiExp_).transpose(),
         this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */delta_))
     * psiShift;
    //anti-periodic boundary condition
    out.slice(0,0,1)*=-1;
    return (psi-out);
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::MMdagger(const NSL::Tensor<Type> & psi){
    const NSL::size_t Nt = this->phi_.shape(0);
    const NSL::size_t Nx = this->phi_.shape(1);
    
   //MMdagger(psi) = M(psi) + Mdagger(psi) + exp_hopping_mtrix^2 x psi - psi
    NSL::Tensor<Type> out = this->M(psi) 
        + this->Mdagger(psi) 
        + NSL::LinAlg::mat_vec(
            (this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */delta_))*
            (this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */delta_)), 
            NSL::LinAlg::transpose(psi)
        ).transpose();

    return (out-psi);

}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::MdaggerM(const NSL::Tensor<Type> & psi){
    const NSL::size_t Nt = this->phi_.shape(0);
    const NSL::size_t Nx = this->phi_.shape(1);
    

    //MMdagger(psi) = M(psi) + Mdagger(psi) + exp(-i*phi) * (exp_hopping_mtrix^2 x exp(-i*phi)* psi) - psi
    NSL::Tensor<Type> out= this->M(psi) + this->Mdagger(psi) + (NSL::LinAlg::adjoint(this->phiExp_) *
        NSL::LinAlg::mat_vec((this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */delta_))*
         (this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */delta_)),
        ((this->phiExp_ * psi).transpose()))).transpose();

    return (out-psi);

}

//return type
template<NSL::Concept::isNumber Type>
Type NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::logDetM(){
    const int Nt = this->phi_.shape(0);
    const int Nx = this->phi_.shape(1); 

    //For identity matrix 
    const Type length = Nx;
    
    NSL::Tensor<Type> prod(Nt,Nx,Nx);
    NSL::Tensor<Type> out(Nx,Nx);
    Type logdet = 0.0;
    
    prod = this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */this->delta_)* NSL::LinAlg::shift(this->phiExp_,-1).expand(Nx).transpose(1,2);
    //F_{Nt-1}
    out = prod(Nt-1, NSL::Slice(), NSL::Slice());

    //Computing F_{Nt-1}.F_{Nt-2}.....F_0
    for(int t = Nt-2;  t >= 0; t--){
        out.mat_mul(prod(t,NSL::Slice(),NSL::Slice())); 
    }
      
    out += NSL::Matrix::Identity<Type>(Nx);   
    logdet = NSL::LinAlg::logdet(out);

    return logdet;

} 

} // namespace FermionMatrix

#endif //NSL_FERMION_MATRIX_HUBBARD_EXP_TPP
