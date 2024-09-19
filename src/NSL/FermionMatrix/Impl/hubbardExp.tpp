#ifndef NSL_FERMION_MATRIX_HUBBARD_EXP_TPP
#define NSL_FERMION_MATRIX_HUBBARD_EXP_TPP

#include "Lattice/lattice.hpp"
#include "device.tpp"
#include "hubbardExp.hpp"
#include "../../Matrix.hpp"
#include "sliceObj.tpp"

namespace NSL::FermionMatrix {

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::F_(const NSL::Tensor<Type> & psi){
    // We want to compute 
    //        [\exp(δK)]_{xy} \exp(i φ_{iy}) B_t δ_{t,i+1} \psi_{yi}
    // Let us first group things into element-wise multiplications, matrix multiplications, and shifts.
    //        B_t δ_{t,i+1} [\exp(δK)]_{xy} (\exp(i φ_{iy}) \psi_{yi})
    //        |---shift---> |---mat mul---> |--- element-wise mul ---|

    NSL::Tensor<Type> Fpsi = NSL::LinAlg::mat_vec(
    // The needed matrix multiplication is on the spatial index.
        this->Lat.exp_hopping_matrix(sgn_*delta_),
    // To get correct broadcasting we transpose the element-wise multiplication
    // so that each column is Nx big.
        (this->phiExp_ * psi).transpose(-1,-2)
    ).transpose(-1,-2);
    // and then transpose back.

    // Now Fpsi contains
    // [\exp(δK)]_{xy} (\exp(i φ_{iy}) \psi_{yi})
    // What remains is to shift it
    // and apply B
    Fpsi.shift(/*shift*/1,/*dim*/-2,/*boundary*/Type(-1));

    return Fpsi;
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::M(const NSL::Tensor<Type> & psi){
    NSL::Tensor<Type> Fpsi = NSL::LinAlg::mat_mul(
        psi*this->phiExp_,
        this->Lat.exp_hopping_matrix(sgn_*delta_)
    );

    Fpsi.shift(/*shift*/1,/*dim*/-2,/*boundary*/Type(-1));

    return psi - Fpsi;
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::Mdagger(const NSL::Tensor<Type> & psi){
    /** We derive M† as follows:
      *     M_{tx,iy}   = δ_{xy} δ_{ti} - B_t [exp(δΚ)]_{xy}   exp(+iφ_{iy}  ) δ_{t,i+1}
      *     M_{tx,iy}^* = δ_{xy} δ_{ti} - B_t [exp(δΚ)]_{xy}^* exp(-iφ_{iy}^*) δ_{t,i+1}
      *         = (M^*T)_{iy,tx} = (M†)_{iy,tx}
      * so now we just relabel
      *     M†_{tx,iy}  = δ_{yx} δ_{it} - B_i [exp(δΚ)]_{yx}^* exp(-iφ_{tx}^*) δ_{i,t+1}
      * and massage
      *     M†_{tx,iy}  = δ_{xy} δ_{ti} - B_i [exp(δΚ)†]_{xy} exp(-iφ_{tx}^*) δ_{t+1,i}.
      *                 = δ_{xy} δ_{ti} - B_i [exp(δ^* Κ)]_{xy} exp(-iφ_{tx}^*) δ_{t+1,i}.
      * which simplified slightly as K is Hermitian.
      **/

    /** If we now consider applying M† to ψ_{iy} we get
      *     (M†ψ)_{tx}  = ψ_tx - exp(-iφ_{tx}^*)      δ_{t+1,i} B_i     [exp(δ^* K)]_{xy}  ψ_{iy}
      *                                                         |- * -> |--- matrix multiply ---|
      **/

    NSL::Tensor<Type> BexpKpsi = NSL::LinAlg::mat_mul(
        psi,
        this->Lat.exp_hopping_matrix(sgn_*NSL::LinAlg::conj(delta_))
    );
    
    /** We now need to evaluate
      *     (M†ψ)_{tx}  = ψ_tx - exp(-iφ_{tx}^*)      δ_{t+1,i} expKpsi_{ix}
      *     (M†ψ)_{tx}  = ψ_tx - exp(-iφ_{tx}^*)      δ_{t,i-1} expKpsi_{ix}
      *                          |- element-wise * -->|------- shift ------|
      **/
    BexpKpsi.shift(/*shift*/-1,/*dim*/-2,/*boundary*/Type(-1));

    return psi - ( this->phiExpCon_ * BexpKpsi);
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::MMdagger(const NSL::Tensor<Type> & psi){
    return this->M(this->Mdagger(psi));
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::MdaggerM(const NSL::Tensor<Type> & psi){
    return this->Mdagger(this->M(psi));
}

//return type
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
Type NSL::FermionMatrix::HubbardExp<Type,LatticeType>::logDetM(){
    const int Nt = this->phi_.shape(0);
    const int Nx = this->phi_.shape(1); 

    // having a batch dimension requires a bit more work and refactoring of 
    // this algorithm for now we don't implement it here
    assertm( this->phi_.dim() == 2, "NSL::FermionMatrix::HubbardExp::logDetM; phi must be a 2D tensor" );

    NSL::Device device = this->phi_.device();

    NSL::Tensor<Type> prod(device,Nt,Nx,Nx);
    NSL::Tensor<Type> sausage = NSL::Matrix::Identity<Type>(device,Nx);
    
    prod = this->Lat.exp_hopping_matrix(sgn_*this->delta_)* NSL::LinAlg::shift(this->phiExp_,-1).expand(Nx).transpose(1,2);

    //Computing F_{Nt-1}.F_{Nt-2}.....F_0
    for(int t = Nt-1;  t >= 0; t--){
        sausage.mat_mul(prod(t,NSL::Slice(),NSL::Slice())); 
    }
    
    return NSL::LinAlg::logdet(NSL::Matrix::Identity<Type>(device,Nx) + sausage);
} 

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::gradLogDetM(){
    //ToDo: implement
    const int Nt = this->phi_.shape(0);
    const int Nx = this->phi_.shape(1);
    const NSL::Device device = this->phi_.device();

    // having a batch dimension requires a bit more work and refactoring of 
    // this algorithm for now we don't implement it here
    assertm( this->phi_.dim() == 2, "NSL::FermionMatrix::HubbardExp::logDetM; phi must be a 2D tensor" );

    NSL::complex<NSL::RealTypeOf<Type>> II = NSL::complex<NSL::RealTypeOf<Type>> {0,1.0};

    // Fk(t) = exp(i phi_{x,t-1})^{-1} * exp(-k)
    Fk_ =  NSL::LinAlg::shift(this->phiExpInv_,+1).expand(Nx).transpose(1,2).transpose() * this->Lat.exp_hopping_matrix(-1 * sgn_* this->delta_);

    // Computing F_{0}^{-1}.F_{1}^{-1}.....F_{Nt-1}^{-1}
    // FkFkFk(t) = Fk(t).Fk(t+1)....Fk(Nt-1)
    // FkFkFk(t=0) gives A^-1 (see eq. 2.32 of Jan-Lukas' notes in hubbardFermionAction.pdf)
    FkFkFk_(Nt-1,NSL::Slice(),NSL::Slice()) = Fk_(Nt-1,NSL::Slice(),NSL::Slice());  // initialize FkFkFk
    for(int t = Nt-2;  t >=0; t--){
	    FkFkFk_(t,NSL::Slice(),NSL::Slice()) = NSL::LinAlg::mat_mul(
            Fk_(t,NSL::Slice(),NSL::Slice()),
            FkFkFk_(t+1,NSL::Slice(),NSL::Slice())
        );
    }

    // this gives (1+A^-1)^-1  (see eq. 2.31 of Jan-Lukas' notes in hubbardFermionAction.pdf)
    invAp1F_ = NSL::LinAlg::mat_inv(NSL::Matrix::Identity<Type>(device,Nx) 
             + FkFkFk_(0,NSL::Slice(),NSL::Slice()));  

    /**
      * We want to calculate Tr((1+A^-1)^-1 ∂_{xt} A^-1 )
      * This is equal to Tr((1+A^-1)^-1 F_{0}^{-1} F_{1}^{-1} .... F_{t}^{-1})_{i,j} δ_{jx} F_{t+1}^{-1}_{x,k} .... F_{Nt-1}^{-1} ) * i
      * Under the trace we can move the terms cyclicly (is that a real word?)
      *                = Tr( δ_{jx} F_{t+1}^{-1}_{x,k} .... F_{Nt-1}^{-1} (1+A^-1)^-1 F_{0}^{-1} F_{1}^{-1} .... F_{t}^{-1})_{i,j} ) * i
      *                = [ F_{t+1}^{-1} F_{t+2}^{-1} .... F_{Nt-1}^{-1} (1+A^-1)^-1 F_{0}^{-1} F_{1}^{-1} .... F_{t}^{-1}) ]_{x,x} * i
      *
      *                = [FkFkFk(t+1).invAp1.Fk(0).Fk(1)...Fk(t)]_{x,x} * i
      *
      * (Note:  there is no sum over x)
      **/

    // first do t=Nt-1 case
    pi_dot_(Nt-1,NSL::Slice()) = II * NSL::LinAlg::diag(
        NSL::LinAlg::mat_mul(FkFkFk_(0,NSL::Slice(),NSL::Slice()),invAp1F_)
    );

    // now do the other timeslices
    for (int t=0; t < Nt-1; t++) {
        // (1+A^-1)^-1 F_{0}^{-1} F_{1}^{-1} .... F_{t}^{-1})
    	invAp1F_.mat_mul(Fk_(t,NSL::Slice(),NSL::Slice()));                 
    	
        pi_dot_(t,NSL::Slice()) =  II * NSL::LinAlg::diag(
            NSL::LinAlg::mat_mul(FkFkFk_(t+1,NSL::Slice(),NSL::Slice()),invAp1F_)
        );
    }

    return pi_dot_;
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::dMdPhi(const NSL::Tensor<Type> & left, const NSL::Tensor<Type> & right){
    NSL::Tensor<Type> sum = NSL::LinAlg::mat_mul(
        left,
        this->Lat.exp_hopping_matrix(sgn_*delta_)
    );
    
    sum.shift(/*shift*/-1,/*dim*/-2,/*boundary*/Type(-1));
    sum *= this->phiExp_ * right;
    
    return -NSL::complex<NSL::RealTypeOf<Type>>(0,1) * sum;
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::dMdaggerdPhi(const NSL::Tensor<Type> & left, const NSL::Tensor<Type> & right){
    NSL::Tensor<Type> sum = NSL::LinAlg::mat_mul(
        right,
        this->Lat.exp_hopping_matrix(sgn_*delta_)
    );

    sum.shift(/*shift*/-1,/*dim*/-2,/*boundary*/Type(-1));
    sum *= this->phiExpInv_ * left;
    
    return NSL::complex<NSL::RealTypeOf<Type>>(0,1) * sum;
}

} // namespace FermionMatrix

#endif //NSL_FERMION_MATRIX_HUBBARD_EXP_TPP
