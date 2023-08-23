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
    throw std::logic_error("FermionMatrix::U1::Wilson::logDetM is not implemented");
    return 0;
}

template<NSL::Concept::isNumber Type>
NSL::Tensor<Type> NSL::FermionMatrix::U1::Wilson<Type>::gradLogDetM(){
    throw std::logic_error("FermionMatrix::U1::Wilson::gradLogDetM is not implemented");
    return NSL::zeros_like(U_);
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
