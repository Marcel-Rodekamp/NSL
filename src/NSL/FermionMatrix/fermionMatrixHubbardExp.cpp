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
        this->Lat->exp_hopping_matrix(/*delta=(beta/Nt) */0.1),
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
         this->Lat->exp_hopping_matrix(0.1))
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
    //const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);

    out= this->M(psi) + this->Mdagger(psi) + NSL::LinAlg::mat_vec((this->Lat->exp_hopping_matrix(0.1))*
        (this->Lat->exp_hopping_matrix(0.1)), NSL::LinAlg::mat_transpose(psi)).
    transpose();

    return (out-psi);

}

template<typename Type>
NSL::TimeTensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::MdaggerM(const NSL::TimeTensor<Type> & psi){
    NSL::TimeTensor<Type> out;
    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1);
    //const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);


    out= this->M(psi) + this->Mdagger(psi) + (NSL::LinAlg::adjoint(this->phiExp_) *
        NSL::LinAlg::mat_vec((this->Lat->exp_hopping_matrix(0.1))*
         (this->Lat->exp_hopping_matrix(0.1)),
        ((this->phiExp_ * psi).transpose()))).transpose();

    return (out-psi);

}

//return type
template<typename Type>
Type NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::logDetM(){
    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1); 
    

    //for identity matrix
    NSL::Matrices::Matrices<Type> Id; 
    const Type length = Nx;
    
    NSL::TimeTensor<Type> prod(Nt,Nx,Nx);
    NSL::TimeTensor<Type> out(Nx,Nx);
    Type logdet = 0.0;
    
    prod = this->Lat->exp_hopping_matrix(0.1)* NSL::LinAlg::shift(this->phiExp_,-1).expand(Nx).transpose(1,2);
    //F_{N_t-1}
    out = prod.slice(/*dim=*/0,/*start=*/Nt-1,/*end=*/Nt);
    
    for(int t=Nt-2; t>=0; t--){

        //! \todo: figure out what mat_mul does
        //out = NSL::LinAlg::mat_vec(out,prod.slice(/*dim=*/0,/*start=*/t,/*end=*/t+1));
        out.mat_mul(prod.slice(/*dim=*/0,/*start=*/t,/*end=*/t+1)); 
         
    }
      
    out += Id.Identity(Nx);
    
    logdet = NSL::LinAlg::logdet(out);

    return logdet;

}

template<typename Type>
Type NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::logDetMdagger() {

    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1); 
    const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);
    

    //for identity matrix
    NSL::Matrices::Matrices<Type> Id; 
    const Type length = Nx;
    
    NSL::TimeTensor<Type> prod;
    NSL::TimeTensor<Type> Ainv;
    Type logdet(0,0), phiSum(0,0);


    prod = this->Lat->exp_hopping_matrix(0.1)* NSL::LinAlg::shift(this->phiExp_,-1).expand(Nx).transpose(1,2);
    //F_{N_0}
    Ainv = NSL::LinAlg::mat_inv(prod.slice(/*dim=*/0,/*start=*/0,/*end=*/1));
    for(int t=1; t<Nt; t++){
        Ainv.mat_mul(NSL::LinAlg::mat_inv(prod.slice(/*dim=*/0,/*start=*/t,/*end=*/t+1)));
        
    }


    for(int j=0; j<Nt; j++){
        for(int k=0; k<Nx; k++){
            phiSum += this->phi_(j,k); //check if it works
            }
    }
    //! \todo: confirm how to do this!
    if(this->sigma_==1) {
        logdet = NSL::LinAlg::logdet(Ainv + Id.Identity(Nx)) - 
                   (phiSum*I) -Nt*NSL::LinAlg::logdet(this->Lat->exp_hopping_matrix(0.1)); //confirm sign of hopping term
        }
    else  {
        logdet = NSL::LinAlg::logdet(Ainv + Id.Identity(Nx)) - 
               (phiSum*I) -Nt*NSL::LinAlg::logdet(this->Lat->exp_hopping_matrix(-0.1)); //confirm sign of hopping term
         
}            

    return logdet;
   
}

//! \todo: Full support of complex number multiplication is missing in Tensor:
//template class NSL::FermionMatrix::FermionMatrixHubbardExp<float>;
//template class NSL::FermionMatrix::FermionMatrixHubbardExp<double>;
template class NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<float>>;
template class NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<double>>;

} // namespace FermionMatrix
