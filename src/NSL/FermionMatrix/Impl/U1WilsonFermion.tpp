#ifndef NSL_FERMION_MATRIX_WILSON_TPP
#define NSL_FERMION_MATRIX_WILSON_TPP
#include "U1WilsonFermion.hpp"

//! todo define gamma matrices

namespace NSL::FermionMatrix::U1 {


template<NSL::Concept::isNumber Type>
void NSL::FermionMatrix::U1::Wilson<Type>::populate(const NSL::Tensor<Type> & phi){
    // this happens once, when U_ is not initialized. We need to tell
    // that a GPU Tensor is incoming. But this call doesn't actually 
    // copy data!
    if(U_.dim() != phi.dim()){
        U_.reshape( phi.shape() );
    }
    U_ = phi ;
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::U1::Wilson<Type>::M(const NSL::Tensor<Type> & psi){
    // shift(Tensor, shift, dim, boundary);
    NSL::Tensor<NSL::complex<NSL::RealTypeOf<Type>>> out = NSL::zeros_like(psi);
    
    auto Id = NSL::Matrix::Identity<Type>( psi.device(), dim_);

    NSL::size_t mu = 0; // This is the time axis, i.e. anti-periodic boundary
    // -1/2 * (1-gamma_mu)^{alpha,beta} psi(x+mu) U_mu(x)
    out -= 0.5 * NSL::LinAlg::mat_vec(
        NSL::LinAlg::shift(psi,-1,mu,Type(-1)) 
            * U_(NSL::Ellipsis(),NSL::NewDim(),mu),
        (Id-gamma_(mu)).transpose(0,1)
    );

    // -1/2 * (1+gamma_mu)^{alpha,beta} psi(x-mu) U_mu(x-mu)^+
    out -= 0.5 * NSL::LinAlg::mat_vec(
        NSL::LinAlg::shift(psi,1,mu,Type(-1)) 
            * 1./NSL::LinAlg::shift(U_(NSL::Ellipsis(),NSL::NewDim(),mu),1,mu),
        (Id+gamma_(mu)).transpose(0,1)
    );

    // now the spatial part; i.e. periodic boundary
    for(mu = 1; mu < dim_; ++mu){
        // -1/2 * (1-gamma_mu)^{alpha,beta} psi(x+mu) U_mu(x)
        out -= 0.5 * NSL::LinAlg::mat_vec(
            NSL::LinAlg::shift(psi,-1,mu) 
                * U_(NSL::Ellipsis(),NSL::NewDim(),mu),
            (Id-gamma_(mu)).transpose(0,1)
        );

        // -1/2 * (1+gamma_mu)^{alpha,beta} psi(x-mu) U_mu(x-mu)^+
        out -= 0.5 * NSL::LinAlg::mat_vec(
            NSL::LinAlg::shift(psi,1,mu) 
                * 1./NSL::LinAlg::shift(U_(NSL::Ellipsis(),NSL::NewDim(),mu),1,mu),
            (Id+gamma_(mu)).transpose(0,1)
        );
    } // for mu

    // add diagonal term
    out += (bareMass_+dim_) * psi;

    return out;
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::U1::Wilson<Type>::Mdagger(const NSL::Tensor<Type> & psi){
    // shift(Tensor, shift, dim, boundary);
    NSL::Tensor<NSL::complex<NSL::RealTypeOf<Type>>> out = NSL::zeros_like(psi);
    
    auto Id = NSL::Matrix::Identity<Type>( psi.device(), dim_);
   
    NSL::size_t mu = 0; // This is the time axis, i.e. anti-periodic boundary
    // -1/2 * (1-gamma_mu)^{alpha,beta} psi(x-mu) U_mu(x)^+
    out -= 0.5 * NSL::LinAlg::mat_vec(
        NSL::LinAlg::shift(psi,1,mu,Type(-1))
            * NSL::LinAlg::shift(1./U_(NSL::Ellipsis(),NSL::NewDim(),mu),1,mu),
        (Id-gamma_(mu)).transpose(0,-1)
    );

    // -1/2 * (1+gamma_mu)^{alpha,beta} psi(x+mu) U_mu(x-mu)
    out -= 0.5 * NSL::LinAlg::mat_vec(
        NSL::LinAlg::shift(psi,-1,mu,Type(-1))
            * U_(NSL::Ellipsis(),NSL::NewDim(),mu),
        (Id+gamma_(mu)).transpose(0,-1)
    );

    // now the spatial part; i.e. periodic boundary
    // ToDo: We can vectorize these for loops with an NSL::Slice(1) == [1:]
    for(mu = 1; mu < dim_; ++mu){
        // -1/2 * (1-gamma_mu)^{alpha,beta} psi(x-mu) U_mu(x)^+
        out -= 0.5 * NSL::LinAlg::mat_vec(
            NSL::LinAlg::shift(psi,1,mu)
                * NSL::LinAlg::shift(1./U_(NSL::Ellipsis(),NSL::NewDim(),mu),1,mu),
            (Id-gamma_(mu)).transpose(0,-1)
        );

        // -1/2 * (1+gamma_mu)^{alpha,beta} psi(x+mu) U_mu(x-mu)
        out -= 0.5 * NSL::LinAlg::mat_vec(
            NSL::LinAlg::shift(psi,-1,mu) 
                * U_(NSL::Ellipsis(),NSL::NewDim(),mu),
            (Id+gamma_(mu)).transpose(0,-1)
        );

    } // for mu

    // add diagonal term
    out += (bareMass_+dim_) * psi;

    return out;
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::U1::Wilson<Type>::MMdagger(const NSL::Tensor<Type> & psi){
    return M(Mdagger( psi ));
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::U1::Wilson<Type>::MdaggerM(const NSL::Tensor<Type> & psi){
    return Mdagger(M( psi ));
}

template<NSL::Concept::isNumber Type>
Type NSL::FermionMatrix::U1::Wilson<Type>::logDetM(){
    Type kappa = 0.5/(bareMass_+2);
    NSL::size_t Nt = U_.shape(0);
    NSL::size_t Nx = U_.shape(1);

    NSL::Tensor<Type> Ap (2*Nx,2*Nx);
    NSL::Tensor<Type> Am (2*Nx,2*Nx);
    NSL::Tensor<Type> T = NSL::eye<NSL::complex<double>>(2*Nx);
    NSL::Tensor<Type> B (2*Nx,2*Nx);
    NSL::Tensor<Type> C (Nx,Nx);
    NSL::Tensor<Type> D (Nx,Nx);
    Type detR = 1;

    for(NSL::size_t t = 0; t < Nt; t++){
        D = NSL::zeros_like(D);
        C = NSL::zeros_like(C);
        B = NSL::zeros_like(B);
         for(NSL:: size_t x = 0; x < Nx; x++){
            D(x,x) += 1;
            D(x,(x+1)%Nx) += -1*kappa*U_(t,x,1);
            D(x,(x-1+Nx)%Nx) += NSL::LinAlg::conj(-1*kappa*U_(t,(x-1+Nx)%Nx,1));
            C(x,(x+1)%Nx) += kappa*U_(t,x,1);
            C(x,(x-1+Nx)%Nx) += std::conj(-1*kappa*U_(t,(x-1+Nx)%Nx,1));

            Ap(x,x) = U_(t,x,0);
            Ap(x+Nx,x+Nx) = U_(t,x,0);
            Am(x,x) = std::conj(U_((t-1+Nt)%Nt,x,0));
            Am(x+Nx,x+Nx) = std::conj(U_((t-1+Nt)%Nt,x,0));
            
        }
        B(NSL::Slice(0,Nx) ,NSL::Slice(0,Nx)) = D;
        B(NSL::Slice(Nx,Nx*2) ,NSL::Slice(Nx,Nx*2)) = D;
        B(NSL::Slice(0,Nx) ,NSL::Slice(Nx,Nx*2)) = C;
        B(NSL::Slice(Nx,Nx*2) ,NSL::Slice(0,Nx)) = C;
       

        NSL::Tensor<Type> S = NSL::LinAlg::mat_mul(B,P0m)-2*kappa*NSL::LinAlg::mat_mul(Ap,P0p);
        NSL::Tensor<Type> R = NSL::LinAlg::mat_mul(B,P0p)-2*kappa*NSL::LinAlg::mat_mul(Am,P0m);

        NSL::Tensor<Type> Rinv = NSL::LinAlg::mat_inv(R);

        T.mat_mul(NSL::LinAlg::mat_mul(Rinv,S));
        detR *= NSL::LinAlg::det(R);
    
    }
    Type res2 = detR * NSL::LinAlg::det(id+T);
    Type res = NSL::LinAlg::log(detR * NSL::LinAlg::det(id+T)) + Nx*Nt*2*NSL::LinAlg::log(bareMass_+2);
    return res;
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::U1::Wilson<Type>::gradLogDetM(){
    Type kappa = 0.5/(bareMass_+2);
    NSL::complex<double> I{0,1};
    NSL::size_t Nt = U_.shape(0);
    NSL::size_t Nx = U_.shape(1);
    NSL::Tensor<Type> force = NSL::zeros_like(U_);
    NSL::Tensor<Type> Ap (2*Nx,2*Nx);
    NSL::Tensor<Type> Am (2*Nx,2*Nx);
    NSL::Tensor<Type> B (2*Nx,2*Nx);
    NSL::Tensor<Type> C (Nx,Nx);
    NSL::Tensor<Type> D (Nx,Nx);
    NSL:: Tensor <Type> dR(2*Nx, 2*Nx);
    NSL:: Tensor <Type> dS(2*Nx,2*Nx);
    NSL::Tensor<Type> inv_saus(2*Nx,2*Nx);
    NSL::Tensor<Type> dT(2*Nx,2*Nx);
    NSL::Tensor<Type> dTm(2*Nx, 2*Nx);



    for(NSL::size_t t = 0; t < Nt; t++){

        D = NSL::zeros_like(D);
        C = NSL::zeros_like(C);
        B = NSL::zeros_like(B);
         for(NSL:: size_t x = 0; x < Nx; x++){
            D(x,x) += 1;
            D(x,(x+1)%Nx) += -1*kappa*U_(t,x,1);
            D(x,(x-1+Nx)%Nx) += NSL::LinAlg::conj(-1*kappa*U_(t,(x-1+Nx)%Nx,1));
            C(x,(x+1)%Nx) += kappa*U_(t,x,1);
            C(x,(x-1+Nx)%Nx) += std::conj(-1*kappa*U_(t,(x-1+Nx)%Nx,1));

            Ap(x,x) = U_(t,x,0);
            Ap(x+Nx,x+Nx) = U_(t,x,0);
            Am(x,x) = std::conj(U_((t-1+Nt)%Nt,x,0));
            Am(x+Nx,x+Nx) = std::conj(U_((t-1+Nt)%Nt,x,0));
            
        }
        B(NSL::Slice(0,Nx) ,NSL::Slice(0,Nx)) = D;
        B(NSL::Slice(Nx,Nx*2) ,NSL::Slice(Nx,Nx*2)) = D;
        B(NSL::Slice(0,Nx) ,NSL::Slice(Nx,Nx*2)) = C;
        B(NSL::Slice(Nx,Nx*2) ,NSL::Slice(0,Nx)) = C;
       
        NSL::Tensor<Type> S = NSL::LinAlg::mat_mul(B,P0m)-2*kappa*NSL::LinAlg::mat_mul(Ap,P0p);
        NSL::Tensor<Type> R = NSL::LinAlg::mat_mul(B,P0p)-2*kappa*NSL::LinAlg::mat_mul(Am,P0m);
        NSL::Tensor<Type> Rinv = NSL::LinAlg::mat_inv(R);

        NSL::Tensor<Type> dB (2*Nx,2*Nx);
        NSL::Tensor<Type> dC (Nx,Nx);
        NSL::Tensor<Type> dD (Nx,Nx);

        for(NSL:: size_t x = 0; x < Nx; x++){

            dB = NSL::zeros_like(B);
            dD = NSL::zeros_like(D);
            dC = NSL::zeros_like(C);

    
            dD(x,(x+1)%Nx) += -1*kappa*U_(t,x,1)*I;
            dD((x+1)%Nx,x) += NSL::LinAlg::conj(-1*kappa*U_(t,x,1)) * -I;
            dC(x,(x+1)%Nx) += kappa*U_(t,x,1) * I;
            dC((x+1)%Nx,x) += std::conj(-1*kappa*U_(t,x,1)) * -I;

            dB(NSL::Slice(0,Nx) ,NSL::Slice(0,Nx)) = dD;
            dB(NSL::Slice(Nx,Nx*2) ,NSL::Slice(Nx,Nx*2)) = dD;
            dB(NSL::Slice(0,Nx) ,NSL::Slice(Nx,Nx*2)) = dC;
            dB(NSL::Slice(Nx,Nx*2) ,NSL::Slice(0,Nx)) = dC;

            dR = NSL::LinAlg::mat_mul(dB,P0p);
            dS = NSL::LinAlg::mat_mul(dB,P0m);

            NSL::Tensor<Type> dTi = NSL::LinAlg::mat_mul(Rinv, dS) - NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(Rinv, NSL::LinAlg::mat_mul(dR, Rinv)), S);

            inv_saus = NSL::LinAlg::mat_inv(id+TT(0,Nt));



            if (t==0){
                force(t,x,1) = NSL::LinAlg::trace(NSL::LinAlg::mat_mul(Rinv,dR)) + NSL::LinAlg::trace(NSL::LinAlg::mat_mul(inv_saus, NSL::LinAlg::mat_mul(dTi, TT(t+1,Nt))));
            }else if (t == Nt-1){
                force(t,x,1) = NSL::LinAlg::trace(NSL::LinAlg::mat_mul(Rinv,dR)) + NSL::LinAlg::trace(NSL::LinAlg::mat_mul(inv_saus, NSL::LinAlg::mat_mul(TT(0,t),dTi)));
            }else{
                force(t,x,1) = NSL::LinAlg::trace(NSL::LinAlg::mat_mul(Rinv,dR)) + NSL::LinAlg::trace(NSL::LinAlg::mat_mul(inv_saus, NSL::LinAlg::mat_mul(TT(0,t),NSL::LinAlg::mat_mul(dTi,TT(t+1,Nt)))));
            }
        }

        NSL::Tensor<Type> dAp (2*Nx,2*Nx);
        NSL::Tensor<Type> dAm (2*Nx,2*Nx);

        for(NSL:: size_t x = 0; x < Nx; x++){
            dAp = NSL::zeros_like(Ap);
            dAm = NSL::zeros_like(Am);

            dAp(x,x) = U_(t,x,0) *I;
            dAp(x+Nx,x+Nx) = U_(t,x,0) *I;
            dAm(x,x) = std::conj(U_((t-1+Nt)%Nt,x,0)) * -I;
            dAm(x+Nx,x+Nx) = std::conj(U_((t-1+Nt)%Nt,x,0)) * -I;

            dR = -2*kappa * NSL::LinAlg::mat_mul(dAm,P0m);
            dS = -2* kappa * NSL::LinAlg::mat_mul(dAp,P0p);

            dT = NSL::LinAlg::mat_mul(Rinv, dS);

            dTm = - NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(Rinv, NSL::LinAlg::mat_mul(dR, Rinv)),S);

            force((t-1+Nt)%Nt,x,0) += NSL::LinAlg::trace(NSL::LinAlg::mat_mul(Rinv, dR));


            if (t==0){
                force(t,x,0) +=  NSL::LinAlg::trace(NSL::LinAlg::mat_mul(inv_saus, NSL::LinAlg::mat_mul(dT, TT(t+1,Nt))));
                force((t-1+Nt)%Nt,x,0) +=  NSL::LinAlg::trace(NSL::LinAlg::mat_mul(inv_saus, NSL::LinAlg::mat_mul(dTm, TT(t+1,Nt))));
            }else if (t == Nt-1){
                force(t,x,0) += NSL::LinAlg::trace(NSL::LinAlg::mat_mul(inv_saus, NSL::LinAlg::mat_mul(TT(0,t),dT)));
                force((t-1+Nt)%Nt,x,0) +=  NSL::LinAlg::trace(NSL::LinAlg::mat_mul(inv_saus, NSL::LinAlg::mat_mul(TT(0,t),dTm)));
            }else{
                force(t,x,0) +=  NSL::LinAlg::trace(NSL::LinAlg::mat_mul(inv_saus, NSL::LinAlg::mat_mul(TT(0,t),NSL::LinAlg::mat_mul(dT,TT(t+1,Nt)))));
                force((t-1+Nt)%Nt,x,0) +=  NSL::LinAlg::trace(NSL::LinAlg::mat_mul(inv_saus, NSL::LinAlg::mat_mul(TT(0,t),NSL::LinAlg::mat_mul(dTm,TT(t+1,Nt)))));
            }
        }
    }
    return force; //+ Nx*Nt*2*NSL::LinAlg::log(bareMass_ + 2);
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::U1::Wilson<Type>::TT(NSL::size_t i, NSL::size_t j){


    Type kappa = 0.5/(bareMass_+2);
    NSL::size_t Nt = U_.shape(0);
    NSL::size_t Nx = U_.shape(1);

    NSL::Tensor<Type> Ap (2*Nx,2*Nx);
    NSL::Tensor<Type> Am (2*Nx,2*Nx);
    NSL::Tensor<Type> T = NSL::eye<NSL::complex<double>>(2*Nx);
    NSL::Tensor<Type> B (2*Nx,2*Nx);
    NSL::Tensor<Type> C (Nx,Nx);
    NSL::Tensor<Type> D (Nx,Nx);

    for(NSL::size_t t = i; t < j; t++){
        B = NSL::zeros_like(B);
        D = NSL::zeros_like(D);
        C = NSL::zeros_like(C);

         for(NSL:: size_t x = 0; x < Nx; x++){
            D(x,x) += 1;
            D(x,(x+1)%Nx) += -1*kappa*U_(t,x,1);
            D(x,(x-1+Nx)%Nx) += NSL::LinAlg::conj(-1*kappa*U_(t,(x-1+Nx)%Nx,1));
            C(x,(x+1)%Nx) += kappa*U_(t,x,1);
            C(x,(x-1+Nx)%Nx) += std::conj(-1*kappa*U_(t,(x-1+Nx)%Nx,1));

            Ap(x,x) = U_(t,x,0);
            Ap(x+Nx,x+Nx) = U_(t,x,0);
            Am(x,x) = std::conj(U_((t-1+Nt)%Nt,x,0));
            Am(x+Nx,x+Nx) = std::conj(U_((t-1+Nt)%Nt,x,0));
            
        }
        B(NSL::Slice(0,Nx) ,NSL::Slice(0,Nx)) = D;
        B(NSL::Slice(Nx,Nx*2) ,NSL::Slice(Nx,Nx*2)) = D;
        B(NSL::Slice(0,Nx) ,NSL::Slice(Nx,Nx*2)) = C;
        B(NSL::Slice(Nx,Nx*2) ,NSL::Slice(0,Nx)) = C;
       

        NSL::Tensor<Type> S = NSL::LinAlg::mat_mul(B,P0m)-2*kappa*NSL::LinAlg::mat_mul(Ap,P0p);
        NSL::Tensor<Type> R = NSL::LinAlg::mat_mul(B,P0p)-2*kappa*NSL::LinAlg::mat_mul(Am,P0m);

        NSL::Tensor<Type> Rinv = NSL::LinAlg::mat_inv(R);

        T.mat_mul(NSL::LinAlg::mat_mul(Rinv,S));    
    }
  
    return T;
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::U1::Wilson<Type>::dMdPhi(const NSL::Tensor<Type> & left, const NSL::Tensor<Type> & right){
    // shift(Tensor, shift, dim, boundary);
    
    NSL::Tensor<NSL::complex<NSL::RealTypeOf<Type>>> out = NSL::zeros_like(U_);

    NSL::complex<NSL::RealTypeOf<Type>> I{0,1};
    
    auto Id = NSL::Matrix::Identity<Type>( left.device(), dim_);

    // -i/2 * chi_{alpha}(x) (1-gamma_mu)^{alpha,beta} psi_beta(x+mu) U_mu(x)
    NSL::size_t mu = 0; // Temporal part; i.e. anti-periodic boundary
    out(NSL::Ellipsis(),mu) -= NSL::complex<NSL::RealTypeOf<Type>>(0.5)*I * ( 
        left *
        NSL::LinAlg::mat_vec(
            NSL::LinAlg::shift(right,-1,mu,Type(-1)) * U_(NSL::Ellipsis(),NSL::NewDim(), mu),
            (Id-gamma_(mu)).transpose(0,-1)
        ) 
    ).sum(/*dim=*/-1);

    // +i/2 * chi_alpha(x+mu) (1+gamma_mu)^{alpha,beta} psi_beta(x) U_mu(x)^+
    out(NSL::Ellipsis(),mu) += NSL::complex<NSL::RealTypeOf<Type>>(0.5)*I * (
        NSL::LinAlg::shift(left,-1,mu,Type(-1)) *
        NSL::LinAlg::mat_vec(
            right * NSL::LinAlg::conj(U_(NSL::Ellipsis(),NSL::NewDim(),mu)),
            (Id+gamma_(mu)).transpose(0,-1)
        )
    ).sum(/*dim*/-1);

    for(mu = 1; mu < dim_; ++mu){
        // -i/2 * chi_{alpha}(x) (1-gamma_mu)^{alpha,beta} psi_beta(x+mu) U_mu(x)
        out(NSL::Ellipsis(),mu) -= NSL::complex<NSL::RealTypeOf<Type>>(0.5)*I * ( 
            left *
            NSL::LinAlg::mat_vec(
                NSL::LinAlg::shift(right,-1,mu) * U_(NSL::Ellipsis(),NSL::NewDim(), mu),
                (Id-gamma_(mu)).transpose(0,-1)
            ) 
        ).sum(/*dim=*/-1);

        // +i/2 * chi_alpha(x+mu) (1+gamma_mu)^{alpha,beta} psi_beta(x) U_mu(x)^+
        out(NSL::Ellipsis(),mu) += NSL::complex<NSL::RealTypeOf<Type>>(0.5)*I * (
            NSL::LinAlg::shift(left,-1,mu) *
            NSL::LinAlg::mat_vec(
                right * NSL::LinAlg::conj(U_(NSL::Ellipsis(),NSL::NewDim(),mu)),
                (Id+gamma_(mu)).transpose(0,-1)
            )
        ).sum(/*dim*/-1);    
    }

    return out;
}


template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::U1::Wilson<Type>::dMdaggerdPhi(const NSL::Tensor<Type> & left, const NSL::Tensor<Type> & right){
    // shift(Tensor, shift, dim, boundary);
    
    NSL::Tensor<NSL::complex<NSL::RealTypeOf<Type>>> out = NSL::zeros_like(left);

    NSL::complex<NSL::RealTypeOf<Type>> I{0,1};
    
    auto Id = NSL::Matrix::Identity<Type>( left.device(), dim_);

    // -i/2 * chi_{alpha}(x) (1-gamma_mu)^{alpha,beta} psi_beta(x+mu) U_mu(x)
    NSL::size_t mu = 0; // Temporal part; i.e. anti-periodic boundary
    out(NSL::Ellipsis(),mu) += NSL::complex<NSL::RealTypeOf<Type>>(0.5)*I * ( 
        NSL::LinAlg::shift(left,-1,mu,Type(-1)) *
        NSL::LinAlg::mat_vec(
            right * NSL::LinAlg::conj(U_(NSL::Ellipsis(),NSL::NewDim(), mu)),
            (Id-gamma_(mu)).transpose(0,-1)
        ) 
    ).sum(/*dim=*/-1);

    // +i/2 * chi_alpha(x+mu) (1+gamma_mu)^{alpha,beta} psi_beta(x) U_mu(x)^+
    out(NSL::Ellipsis(),mu) -= NSL::complex<NSL::RealTypeOf<Type>>(0.5)*I * (
        left *
        NSL::LinAlg::mat_vec(
            NSL::LinAlg::shift(right,-1,mu,Type(-1)) * U_(NSL::Ellipsis(),NSL::NewDim(),mu),
            (Id+gamma_(mu)).transpose(0,-1)
        )
    ).sum(/*dim*/-1);

    for(mu = 1; mu < dim_; ++mu){
        // -i/2 * chi_{alpha}(x) (1-gamma_mu)^{alpha,beta} psi_beta(x+mu) U_mu(x)
        out(NSL::Ellipsis(),mu) += NSL::complex<NSL::RealTypeOf<Type>>(0.5)*I * ( 
            NSL::LinAlg::shift(left,-1,mu) *
            NSL::LinAlg::mat_vec(
                right * NSL::LinAlg::conj(U_(NSL::Ellipsis(),NSL::NewDim(), mu)),
                (Id-gamma_(mu)).transpose(0,-1)
            ) 
        ).sum(/*dim=*/-1);

        // +i/2 * chi_alpha(x+mu) (1+gamma_mu)^{alpha,beta} psi_beta(x) U_mu(x)^+
        out(NSL::Ellipsis(),mu) -= NSL::complex<NSL::RealTypeOf<Type>>(0.5)*I * (
            left *
            NSL::LinAlg::mat_vec(
                NSL::LinAlg::shift(right,-1,mu) * U_(NSL::Ellipsis(),NSL::NewDim(),mu),
                (Id+gamma_(mu)).transpose(0,-1)
            )
        ).sum(/*dim*/-1);
    }

    return out;
}

} // namespace FermionMatrix

#endif //NSL_FERMION_MATRIX_HUBBARD_EXP_TPP
