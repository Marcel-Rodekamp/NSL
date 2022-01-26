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
    const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);

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
    const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);

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
    const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);


    out= this->M(psi) + this->Mdagger(psi) + (NSL::LinAlg::adjoint(this->phiExp_) *
        NSL::LinAlg::mat_vec((this->Lat->exp_hopping_matrix(0.1))*
         (this->Lat->exp_hopping_matrix(0.1)),
        (NSL::LinAlg::mat_transpose(this->phiExp_)))).transpose();

    return (out-psi);

}

//return type
template<typename Type>
NSL::complex<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::logDetM(const NSL::TimeTensor<Type> & psi){
    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1); 
    

    //for identity matrix
    NSL::Matrices::Matrices<Type> Id; 
    const Type length = Nx;
    
    NSL::TimeTensor<Type> prod;
    NSL::TimeTensor<Type> out;
    NSL::complex<Type> logdet(0,0);
    //F_{N_t-1}
    prod = this->Lat->exp_hopping_matrix(0.1)* NSL::LinAlg::shift(this->phiExp_,-1).expand(Nx).transpose(1,2);
    
    out = prod.slice(/*dim=*/0,/*start=*/Nt-1,/*end=*/Nt);
    
    for(int i=Nt-2; i<=0; i++){
        out = out * prod.slice(/*dim=*/0,/*start=*/i,/*end=*/i+1);    
    }
    
    
    out = out + Id.Identity(Nx);
    std::cout<<out<<std::endl;
    logdet = NSL::LinAlg::logdet(out);

    return logdet;

}

template<typename Type>
NSL::complex<double> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::logDetMdagger(const NSL::TimeTensor<Type> & psi) {

    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1); 
    const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);
    

    //for identity matrix
    NSL::Matrices::Matrices<Type> Id; 
    const Type length = Nx;
    
    NSL::TimeTensor<Type> prod;
    NSL::TimeTensor<Type> Ainv;
    NSL::complex<double> logdet(0,0), phiSum(0,0);
    //F_{N_t-1}

//    prod = this->Lat->exp_hopping_matrix(0.1)* NSL::LinAlg::shift(this->phiExp_,-1).expand(Nx).transpose(1,2);
//    Ainv = NSL::LinAlg::mat_inv(prod.slice(/*dim=*/0,/*start=*/0,/*end=*/1));
//    for(int i=1; i<Nt; i--){
//        Ainv = Ainv * NSL::LinAlg::mat_inv(prod.slice(/*dim=*/0,/*start=*/i,/*end=*/i));
        
//    }


//    for(int j=0; j<Nt; j++){
//        for(int k=0; k<Nx; k++){
//            phiSum=phiSum + this->phi_(j,k); //check if it works
//            }
//    }
//    logdet = NSL::LinAlg::logdet(Ainv + Id.Identity(Nx)) - 
//               (phiSum*I) -Nt*NSL::LinAlg::logdet(this->Lat->exp_hopping_matrix(0.1)); //confirm sign of hopping term
                

    return logdet;
   
}

/*
template<typename Type>
NSL::TimeTensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::F(){
    NSL::TimeTensor<Type> out;
    const NSL::complex<typename RT_extractor<Type>::value_type> I(0,1);
    NSL::TimeTensor<Type> phiShift = NSL::LinAlg::shift(this->phi_,-1);

    out=NSL::LinAlg::mat_vec(this->Lat->exp_hopping_matrix(0.1),
        NSL::LinAlg::mat_transpose(NSL::LinAlg::exp(phi*I)));

    return(out);
}
*/
/*
template<typename Type>
NSL::complex<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::logDetM(const NSL::TimeTensor<Type> & psi){
    const std::size_t Nt = this->phi_.shape(0);
    const std::size_t Nx = this->phi_.shape(1);  
    
    NSL::TimeTensor<Type> prod;
    NSL::complex<Type> out;
    //F_{N_t-1}
    prod = this->Lat->exp_hopping_matrix(0.1)* NSL::LinAlg::shift(this->phiExp_,-(1))
    for(int i=0; i<Nt-2; i++){
        
    }

}
*/
//! \todo: Full support of complex number multiplication is missing in Tensor:
//template class NSL::FermionMatrix::FermionMatrixHubbardExp<float>;
//template class NSL::FermionMatrix::FermionMatrixHubbardExp<double>;
template class NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<float>>;
template class NSL::FermionMatrix::FermionMatrixHubbardExp<NSL::complex<double>>;

} // namespace FermionMatrix
