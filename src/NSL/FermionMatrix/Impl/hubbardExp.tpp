#ifndef NSL_FERMION_MATRIX_HUBBARD_EXP_TPP
#define NSL_FERMION_MATRIX_HUBBARD_EXP_TPP

#include "Lattice/lattice.hpp"
#include "hubbardExp.hpp"
#include "../../Matrix.hpp"

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
        this->Lat.exp_hopping_matrix(delta_),
    // To get correct broadcasting we transpose the element-wise multiplication
    // so that each column is Nx big.
        (this->phiExp_ * psi).transpose()
    ).transpose();
    // and then transpose back.

    // Now Fpsi contains
    // [\exp(δK)]_{xy} (\exp(i φ_{iy}) \psi_{yi})
    // What remains is to shift it
    Fpsi.shift(0,1);
    // and apply B
    Fpsi(0,NSL::Slice()) *= -1;

    return Fpsi;
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::M(const NSL::Tensor<Type> & psi){
    return psi - this->F_(psi);
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
    NSL::Tensor<Type> BexpKpsi = NSL::LinAlg::mat_vec(
        this->Lat.exp_hopping_matrix(NSL::conj(delta_)),
        NSL::LinAlg::transpose(psi)
        ).transpose();
    BexpKpsi(0, NSL::Slice()) *= -1;

    /** We now need to evaluate
      *     (M†ψ)_{tx}  = ψ_tx - exp(-iφ_{tx}^*)      δ_{t+1,i} expKpsi_{ix}
      *     (M†ψ)_{tx}  = ψ_tx - exp(-iφ_{tx}^*)      δ_{t,i-1} expKpsi_{ix}
      *                          |- element-wise * -->|------- shift ------|
      **/

    return psi - ( NSL::conj(this->phiExp_) * (BexpKpsi.shift(0,-1)));
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::MMdagger(const NSL::Tensor<Type> & psi){
    const NSL::size_t Nt = this->phi_.shape(0);
    const NSL::size_t Nx = this->phi_.shape(1);
    
   //MMdagger(psi) = M(psi) + Mdagger(psi) + exp_hopping_mtrix^2 x psi - psi
    NSL::Tensor<Type> out = this->M(psi) 
        + this->Mdagger(psi) 
        + NSL::LinAlg::mat_vec(
            (this->Lat.exp_hopping_matrix(2*delta_)), 
            NSL::LinAlg::transpose(psi)
        ).transpose();

    return (out-psi);

}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::MdaggerM(const NSL::Tensor<Type> & psi){
    const NSL::size_t Nt = this->phi_.shape(0);
    const NSL::size_t Nx = this->phi_.shape(1);
    

    //MMdagger(psi) = M(psi) + Mdagger(psi) + exp(-i*phi) * (exp_hopping_mtrix^2 x exp(-i*phi)* psi) - psi
    NSL::Tensor<Type> out= this->M(psi) + this->Mdagger(psi) + (NSL::LinAlg::adjoint(this->phiExp_) *
        NSL::LinAlg::mat_vec((this->Lat.exp_hopping_matrix(2*delta_)),
        ((this->phiExp_ * psi).transpose()))).transpose();

    return (out-psi);

}

//return type
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
Type NSL::FermionMatrix::HubbardExp<Type,LatticeType>::logDetM(){
    const int Nt = this->phi_.shape(0);
    const int Nx = this->phi_.shape(1); 

    //For identity matrix 
    const Type length = Nx;
    
    NSL::Tensor<Type> prod(Nt,Nx,Nx);
    NSL::Tensor<Type> out(Nx,Nx);
    Type logdet = 0.0;
    
    prod = this->Lat.exp_hopping_matrix(/*delta=(beta/Nt) */this->delta_)* NSL::LinAlg::shift(this->phiExp_,-1).expand(Nx).transpose(1,2);
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
