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
    Fpsi.shift(1,0);
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

    //!ToDo: why do we conjugate delta?  
    NSL::Tensor<Type> BexpKpsi = NSL::LinAlg::mat_vec(
        this->Lat.exp_hopping_matrix(NSL::LinAlg::conj(delta_)),
        NSL::LinAlg::transpose(psi)
        ).transpose();
    BexpKpsi(0, NSL::Slice()) *= -1;

    /** We now need to evaluate
      *     (M†ψ)_{tx}  = ψ_tx - exp(-iφ_{tx}^*)      δ_{t+1,i} expKpsi_{ix}
      *     (M†ψ)_{tx}  = ψ_tx - exp(-iφ_{tx}^*)      δ_{t,i-1} expKpsi_{ix}
      *                          |- element-wise * -->|------- shift ------|
      **/

    return psi - ( NSL::LinAlg::conj(this->phiExp_) * (BexpKpsi.shift(-1,0)));
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::MMdagger(const NSL::Tensor<Type> & psi){
    /** Let's evaluate MM† using the index representations above.
      *     (MM†)_{tx,iy}   = M_{tx,uz} (M†)_{uz,iy}
      *                     = (δ_{tu} δ_{xz} - [\exp(δK)]_{xz} \exp(i φ_{uz}) B_t δ_{t,u+1} ) 
      *                       (δ_{ui} δ_{zy} - B_i [exp(δ^* Κ)]_{zy} exp(-iφ_{uz}^*) δ_{u+1,i})
      * Note that the first term in each paren is just the identity matrix.
      * So, if we expand the parentheses we can write 
      *     (MM†)_{tx,iy}   = M_{tx,iy} + M†_{tx,iy) - δ_{ti} δ_{xy}
      *                     + [\exp(δK)]_{xz} exp(i φ_{uz}) B_t δ_{t,u+1} B_i [exp(δ^* K)]_{zy} exp(-iφ_{uz}^*) δ_{u+1,i}
      *                     = (M + M† - 1)_{tx,iy} + B_t B_i δ_{t,u+1} δ_{u+1,i} [exp(δK)]_{xz} [exp(δ^* Κ)]_{zy} exp(i (φ-φ^*)_{uz}) 
      *                     = (M + M† - 1)_{tx,iy} + B_t B_i δ_{t,i} [exp(δK)]_{xz}  exp(i (φ-φ^*)_{i-1,z}) [exp(δ^* K)]_{zy}
      *                     = (M + M† - 1)_{tx,iy} + (B_t)^2 δ_{t,i} [exp(δK)]_{xz}  exp(i (φ-φ^*)_{i-1,z}) [exp(δ^* K)]_{zy}
      *                     = (M + M† - 1)_{tx,iy} + [exp(δK)]_{xz} δ_{t,i} exp(i (φ-φ^*)_{i-1,z}) [exp(δ^* K)]_{zy}
      *
      * In the case that phi is real this simplifies because the φ-dependent term is 1 and one finds
      *                     = (M + M† - 1)_{tx,iy} + [exp((δ+δ^*)K)]_{xy}
      **/
    return (this->M(psi) + this->Mdagger(psi) - psi) + NSL::LinAlg::mat_vec(
        this->Lat.exp_hopping_matrix(delta_),
        (   NSL::LinAlg::shift(this->phiExp_ * NSL::LinAlg::conj(this->phiExp_), +1, 0)
          * NSL::LinAlg::mat_vec(
                this->Lat.exp_hopping_matrix(NSL::LinAlg::conj(delta_)),
                NSL::LinAlg::transpose(psi)
            ).transpose()
        ).transpose()
    ).transpose();

}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::MdaggerM(const NSL::Tensor<Type> & psi){
    /** Let's evaluate MM† using the index representations above.
      *     (M†M)_{tx,iy}   = (M†)_{tx,uz} M_{uz,iy}
      *                     = (δ_{tu} δ_{xz} - B_u [exp(δ^* K)]_{xz} exp(-iφ_{tx}^*) δ_{t+1,u})
      *                       (δ_{ui} δ_{zy} - B_u δ_{u,i+1} [exp(δK)]_{zy} exp(+iφ_{iy}) )
      * Note that the first term in each paren is just the identity matrix.
      * So, if we expand the parentheses we can write 
      *     (MM†)_{tx,iy}   = M_{tx,iy} + M†_{tx,iy) - δ_{ti} δ_{xy}
      *                     + B_u^2 δ_{t+1,u} δ_{u,i+1} exp(-iφ_{tx}^*) [exp(δ^* Κ)]_{xz} [exp(δ K)]_{zy} exp(+iφ_{iy})
      *                     = (M + M† - 1)_{tx,iy} + δ_{t+1,i+1} exp(-iφ_{tx}^*) [exp((δ^* + δ) Κ)]_{xy} exp(+iφ_{iy})
      *                     = (M + M† - 1)_{tx,iy} + δ_{t,i} exp(-iφ_{tx}^*) [exp((δ^* + δ) Κ)]_{xy} exp(+iφ_{iy})
      *                     = (M + M† - 1)_{tx,iy} + exp(-iφ_{ix}^*) [exp((δ^* + δ) Κ)]_{xy} exp(+iφ_{iy})
      *
      **/
    return this->M(psi) + this->Mdagger(psi) - psi + NSL::LinAlg::conj(this->phiExp_) * NSL::LinAlg::mat_mul(
        this->Lat.exp_hopping_matrix(NSL::LinAlg::conj(delta_)+delta_),
        (this->phiExp_ * psi ).transpose()
    ).transpose();
}

//return type
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
Type NSL::FermionMatrix::HubbardExp<Type,LatticeType>::logDetM(){
    const int Nt = this->phi_.shape(0);
    const int Nx = this->phi_.shape(1); 

    NSL::Tensor<Type> prod(Nt,Nx,Nx);
    NSL::Tensor<Type> sausage = NSL::Matrix::Identity<Type>(Nx);
    
    prod = this->Lat.exp_hopping_matrix(this->delta_)* NSL::LinAlg::shift(this->phiExp_,-1).expand(Nx).transpose(1,2);

    //Computing F_{Nt-1}.F_{Nt-2}.....F_0
    for(int t = Nt-1;  t >= 0; t--){
        sausage.mat_mul(prod(t,NSL::Slice(),NSL::Slice())); 
    }
    
    return NSL::LinAlg::logdet(NSL::Matrix::Identity<Type>(Nx) + sausage);
} 

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::gradLogDetM(){
    //ToDo: implement
    const int Nt = this->phi_.shape(0);
    const int Nx = this->phi_.shape(1);

    NSL::Tensor<Type> Fk(Nt,Nx,Nx);
    NSL::Tensor<Type> FkFkFk(Nt,Nx,Nx);
    NSL::Tensor<Type> invAp1F = NSL::Matrix::Identity<Type>(Nx);
    NSL::Tensor<Type> pi_dot(Nt,Nx);

    // Fk(t) = exp(-i phi_{x,t-1})*exp(-k)
    Fk =  NSL::LinAlg::shift(NSL::LinAlg::conj(this->phiExp_),+1).expand(Nx).transpose(1,2).transpose() * this->Lat.exp_hopping_matrix(-1 * this->delta_);
    
    // Computing F_{0}^{-1}.F_{1}^{-1}.....F_{Nt-1}^{-1}
    // FkFkFk(t) = Fk(t).Fk(t+1)....Fk(Nt-1)
    // FkFkFk(t=0) gives A^-1 (see eq. 2.32 of Jan-Lukas' notes in hubbardFermionAction.pdf)
    FkFkFk(Nt-1,NSL::Slice(),NSL::Slice()) = Fk(Nt-1,NSL::Slice(),NSL::Slice());  // initialize FkFkFk
    for(int t = Nt-2;  t >=0; t--){
	FkFkFk(t,NSL::Slice(),NSL::Slice()) = NSL::LinAlg::mat_mul(Fk(t,NSL::Slice(),NSL::Slice()),FkFkFk(t+1,NSL::Slice(),NSL::Slice()));
    }

    invAp1F = NSL::LinAlg::mat_inv(NSL::Matrix::Identity<Type>(Nx) + FkFkFk(0,NSL::Slice(),NSL::Slice()));  // this gives (1+A^-1)^-1  (see eq. 2.31 of Jan-Lukas' notes in hubbardFermionAction.pdf)

    NSL::Tensor<Type> pi  = NSL::Matrix::Identity<Type>(Nx);
    NSL::complex<NSL::RealTypeOf<Type>> II = NSL::complex<NSL::RealTypeOf<Type>> {0,1.0};

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
    pi = NSL::LinAlg::mat_mul(FkFkFk(0,NSL::Slice(),NSL::Slice()),invAp1F);
    for (int i = 0; i < Nx; i++) {
    	pi_dot(Nt-1,i) =  II * pi(i,i); 	    
    }

    // now do the other timeslices
    for (int t=0; t < Nt-1; t++) {
    	invAp1F.mat_mul(Fk(t,NSL::Slice(),NSL::Slice()));                 // (1+A^-1)^-1 F_{0}^{-1} F_{1}^{-1} .... F_{t}^{-1})
	pi = NSL::LinAlg::mat_mul(FkFkFk(t+1,NSL::Slice(),NSL::Slice()),invAp1F);
    	for (int i= 0; i < Nx; i++) {
	    pi_dot(t,i) =  II * pi(i,i); 	    
	}
    }

    return pi_dot;
}

} // namespace FermionMatrix

#endif //NSL_FERMION_MATRIX_HUBBARD_EXP_TPP
