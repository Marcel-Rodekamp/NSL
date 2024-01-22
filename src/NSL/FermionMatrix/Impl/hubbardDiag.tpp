#ifndef NSL_FERMION_MATRIX_HUBBARD_DIAG_TPP
#define NSL_FERMION_MATRIX_HUBBARD_DIAG_TPP

#include "Lattice/lattice.hpp"
#include "hubbardDiag.hpp"
#include "../../Matrix.hpp"

namespace NSL::FermionMatrix {

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardDiag<Type,LatticeType>::M(const NSL::Tensor<Type> & psi){
    // We want to compute 
    //        \psi_{xt} - [(δK)]_{xy} \psi_{yt} -  \exp(i φ_{iy}) B_t δ_{t,i+1} δ_{xy} \psi_{yi}
    // Let us first group things into element-wise multiplications, matrix multiplications, and shifts.
    //        \psi_{xt} - [(δK)]_{xy} \psi_{yt} - B_t δ_{t,i+1} (\exp(i φ_{ix}) \psi_{xi})
    //                    |---mat mul-------->|   |---shift--->|--- element-wise mul ---|


    NSL::Tensor<Type> out = (this->phiExp_ * psi);
    out.shift(1);
    // and apply B
    out(0,NSL::Slice()) *= -1;

    out += NSL::LinAlg::mat_vec(
    // The needed matrix multiplication is on the spatial index.
        this->Lat.hopping_matrix(sgn_*delta_),
    // To get correct broadcasting we transpose the element-wise multiplication
    // so that each column is Nx big.
        NSL::LinAlg::transpose(psi)
    ).transpose();
    // and then transpose back.

    // Now, we have
    // [(δK)]_{xy} \psi_{yt} +  \exp(i φ_{iy}) B_t δ_{t,i+1} δ_{xy} \psi_{yi}

    return psi - out;
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardDiag<Type,LatticeType>::Mdagger(const NSL::Tensor<Type> & psi){
    
    /** We derive M† as follows:
      *     M_{tx,iy}   = δ_{xy} δ_{ti} - [δΚ]_{xy}δ_{t,i}  - B_t exp(+iφ_{iy}  ) δ_{xy} δ_{t,i+1}
      *     M_{tx,iy}^* = δ_{xy} δ_{ti} - [δ^* Κ*]_{xy}δ_{t,i}  - B_t exp(-iφ_{iy}^* ) δ_{xy} δ_{t,i+1}
      *         = (M^*T)_{iy,tx} = (M†)_{iy,tx}
      * so now we just relabel
      *     M†_{tx,iy}  = δ_{yx} δ_{it} - [δ^* Κ*]_{yx}δ_{t,i}  - B_i exp(-iφ_{tx}^* ) δ_{yx} δ_{t+1,i}
      * and massage
      *     M†_{tx,iy}  = δ_{xy} δ_{ti} - [δ Κ]†_{xy}δ_{t,i}  - B_i exp(-iφ_{tx}^* ) δ_{yx} δ_{t+1,i}
      *                 = δ_{xy} δ_{ti} - [δ^* Κ]_{xy}δ_{t,i}  - B_i exp(-iφ_{tx}^* ) δ_{yx} δ_{t+1,i}.
      * which simplified slightly as K is Hermitian.
      **/

    /** If we now consider applying M† to ψ_{iy} we get
      *     (M†ψ)_{tx}  = ψ_tx - [δ^* Κ]_{xy}δ_{t,i} ψ_{ix} -       exp(-iφ_{tx}^*) δ_{t+1,i} B_i  ψ_{ix}
      *                           |---mat mul-------->|            |element-wise mul|-----shift------>|                                             
      **/

    NSL::Tensor<Type> Bexpconjphipsi(psi,true);

    Bexpconjphipsi(0,NSL::Slice()) *= -1;
    Bexpconjphipsi = NSL::LinAlg::conj(this->phiExp_) * Bexpconjphipsi.shift(-1);
    
    return psi - NSL::LinAlg::mat_vec(
        this->Lat.hopping_matrix(sgn_*NSL::LinAlg::conj(delta_)), 
        NSL::LinAlg::transpose(psi)
    ).transpose() - Bexpconjphipsi;
}


template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardDiag<Type,LatticeType>::MMdagger(const NSL::Tensor<Type> & psi){

    /** Let's evaluate MM† using the index representations above.
      *     (MM†)_{tx,iy}   = M_{tx,uz} (M†)_{uz,iy}
      *                     = (δ_{xz} δ_{tu} - [δΚ]_{xz}δ_{t,u}  - B_t exp(+iφ_{uz} ) δ_{xz} δ_{t,u+1} ) 
      *                       (δ_{zy} δ_{ui} - [δ^* Κ]_{zy}δ_{u,i}  - B_i exp(-iφ_{uz}^* ) δ_{yz} δ_{u+1,i})
      * Note that the first term in each parentheses is just the identity matrix.
      * So, if we expand the parentheses we can write 
      *     (MM†)_{tx,iy}   = M_{tx,iy} + M†_{tx,iy} -1_{tx,iy} + [δΚ]_{xz}δ_{t,u} [δ^* Κ]_{zy}δ_{u,i} + [δΚ]_{xz}δ_{t,u} B_i exp(-iφ_{uz}^* ) δ_{yz} δ_{u+1,i}
      *                     + B_t exp(+iφ_{uz} ) δ_{xz} δ_{t,u+1} [δ^* Κ]_{zy}δ_{u,i} + B_t exp(+iφ_{uz} ) δ_{xz} δ_{t,u+1} B_i exp(-iφ_{uz}^* ) δ_{yz} δ_{u+1,i}
      *If we now consider applying MM† to ψ_{iy} we get
      *      (MM†ψ)_{tx} = Mψ_{tx} + M†ψ_{tx} - ψ{tx} + [δΚ]_{xz} [δ^* Κ]_{zy}δ_{t,i}ψ_{iy} + [δΚ]_{xy} exp(-iφ_{ty}^* ) δ_{i,t+1} B_i ψ_{iy}
      *                                                            |------mat_mul-------->|             |-element-wise mu->l|-----shift------>|
      *                                                 |-------------mat_mul------------>|   |------------------mat_mul------------------->|
      *                   + B_t δ_{t,i+1} exp(+iφ_{ix} ) [δ^* Κ]_{xy}ψ_{iy} +   exp(i (φ-φ^*)_{t-1,x}ψ_{tx}
      *                      |--shift-->||-element-wise mul->|----mat_mul--->|  |-element-wise mul->|
      **/
    //! \todo: Change the names of the variables and check if this can be implemented in a more efficient way

    NSL::Tensor<Type> out, KexpconjphiBpsi, BexpconjdelKpsi, out_;
  
    out = this->M(psi) + this->Mdagger(psi) - psi 
        + NSL::LinAlg::mat_vec(
            this->Lat.hopping_matrix(sgn_*delta_), 
            NSL::LinAlg::mat_vec(
                this->Lat.hopping_matrix(sgn_*NSL::LinAlg::conj(delta_)),NSL::LinAlg::transpose(psi)
            )
    ).transpose();
    
    KexpconjphiBpsi = psi;
    KexpconjphiBpsi(0,NSL::Slice()) *= -1;
    out += NSL::LinAlg::mat_vec(
        this->Lat.hopping_matrix(sgn_*delta_), 
        (NSL::LinAlg::conj(this->phiExp_) * KexpconjphiBpsi.shift(-1)).transpose()
    ).transpose();
 
    BexpconjdelKpsi = (this->phiExp_) * NSL::LinAlg::mat_vec(
        this->Lat.hopping_matrix(sgn_*NSL::LinAlg::conj(delta_)), NSL::LinAlg::transpose(psi)
    ).transpose();
    
    BexpconjdelKpsi.shift(1);
    BexpconjdelKpsi(0,NSL::Slice()) *= -1;
    
    out += BexpconjdelKpsi + (((this->phiExp_) * NSL::LinAlg::conj(this->phiExp_)).shift(1)) * psi;
    
    return out;
  
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardDiag<Type,LatticeType>::MdaggerM(const NSL::Tensor<Type> & psi){

     /** Let's evaluate M†M using the index representations above.
      *     (M†M)_{tx,iy}   = M†_{tx,uz} (M)_{uz,iy}
      *                     = (δ_{xz} δ_{tu} - [δ^* Κ]_{xz}δ_{t,u}  - B_u exp(-iφ_{tx}^* ) δ_{zx} δ_{t+1,u} ) 
      *                       (δ_{zy} δ_{ui} - [δ Κ]_{zy}δ_{u,i}  - B_u exp(iφ_{iy}^* ) δ_{zy} δ_{u,i+1})
      * Note that the first term in each parentheses is just the identity matrix.
      * So, if we expand the parentheses we can write 
      *     (M†M)_{tx,iy}   = M_{tx,iy} + M†_{tx,iy} -1_{tx,iy} + [δ^* Κ]_{xz}δ_{t,u}[δ Κ]_{zy}δ_{u,i} + [δ^* Κ]_{xz}δ_{t,u} B_u exp(iφ_{iy}^* ) δ_{zy} δ_{u,i+1}
      *                     + B_u exp(-iφ_{tx}^* ) δ_{zx} δ_{t+1,u}[δ Κ]_{zy}δ_{u,i} + B_u exp(-iφ_{tx}^* ) δ_{zx} δ_{t+1,u}B_u exp(iφ_{iy}^* ) δ_{zy} δ_{u,i+1}
      *If we now consider applying MM† to ψ_{iy} we get
      *     (M†Mψ)_{tx} = Mψ_{tx} + M†ψ_{tx} - ψ{tx} + [δΚ]_{xz} [δ^* Κ]_{zy}δ_{t,i}ψ_{iy} + B_t δ_{t,i+1} [δ^* Κ]_{xy} exp(+iφ_{iy} ) ψ_{iy}
      *                   exp(-iφ_{tx}^* ) [δΚ]_{xy}  δ_{i,t+1} B_i ψ_{iy} + exp(i (φ-φ^*)_{t,x}ψ_{tx}
      **/

    NSL::Tensor<Type> out, KexpconjphiBpsi, BexpconjdelKpsi, out_;
    out = this->M(psi) + this->Mdagger(psi) - psi 
        + NSL::LinAlg::mat_vec(
            this->Lat.hopping_matrix(sgn_*delta_), 
            NSL::LinAlg::mat_vec(
                this->Lat.hopping_matrix(sgn_*NSL::LinAlg::conj(delta_)),
                NSL::LinAlg::transpose(psi)
            )
        ).transpose();

    KexpconjphiBpsi = psi;
    KexpconjphiBpsi(0,NSL::Slice()) *= -1;
    out += NSL::LinAlg::conj(this->phiExp_) * NSL::LinAlg::mat_vec(
        this->Lat.hopping_matrix(sgn_*delta_), 
        KexpconjphiBpsi.shift(-1).transpose()
    ).transpose();
  
    BexpconjdelKpsi =   NSL::LinAlg::mat_vec(
        this->Lat.hopping_matrix(sgn_*NSL::LinAlg::conj(delta_)), 
        ((this->phiExp_) * psi).transpose()
    ).transpose();
    
    BexpconjdelKpsi.shift(1);
    BexpconjdelKpsi(0,NSL::Slice()) *= -1;
    
    out += BexpconjdelKpsi + ((this->phiExp_) * NSL::LinAlg::conj(this->phiExp_)) * psi;

  return (out);

}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
Type NSL::FermionMatrix::HubbardDiag<Type,LatticeType>::logDetM(){
    const int Nt = this->phi_.shape(0);
    const int Nx = this->phi_.shape(1); 
    const NSL::Device device = this->phi_.device();

    NSL::Tensor<Type> prod(device,Nt,Nx,Nx);
    NSL::Tensor<Type> sausage = NSL::Matrix::Identity<Type>(device,Nx);
    NSL::Tensor<Type> fk(device,Nx,Nx);
    NSL::complex<NSL::RealTypeOf<Type>> II={0,1};
    
    //F^{-1} in expanded form    
    prod = (NSL::LinAlg::shift(phiExpInv_,-1).expand(Nx).transpose(1,2));
    for(int t = 0;  t < Nt ; t++){
        //mat-vec of F^{-1} and K
        fk = prod(t,NSL::Slice(),NSL::Slice()) 
           * (NSL::Matrix::Identity<Type>(device,Nx)-this->Lat.hopping_matrix(sgn_*delta_));
        sausage.mat_mul(fk); 
    }
    //sum over all the elements of phi 
    Type sum = II*sgn_*(this->phi_).sum(); 
  
    return sum + NSL::LinAlg::logdet(NSL::Matrix::Identity<Type>(device,Nx) + sausage);

} 


template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardDiag<Type,LatticeType>::gradLogDetM(){
    //ToDo: implement
    const int Nt = this->phi_.shape(0);
    const int Nx = this->phi_.shape(1);
    const NSL::Device device = this->phi_.device();

    NSL::complex<NSL::RealTypeOf<Type>> II = NSL::complex<NSL::RealTypeOf<Type>> {0,1.0};

    // Fk(t) = exp(i phi_{x,t-1})^{-1} * (1 -k)
    Fk_ =  NSL::LinAlg::shift(this->phiExpInv_,+1).expand(Nx).transpose(1,2).transpose() 
        *( NSL::Matrix::Identity<Type>(device,Nx) - this->Lat.hopping_matrix(sgn_*delta_));

    // Computing F_{0}^{-1}.F_{1}^{-1}.....F_{Nt-1}^{-1}
    // FkFkFk(t) = Fk(t).Fk(t+1)....Fk(Nt-1)
    // FkFkFk(t=0) gives A^-1 (see eq. 2.32 of Jan-Lukas' notes in hubbardFermionAction.pdf)
    FkFkFk_(Nt-1,NSL::Slice(),NSL::Slice()) = Fk_(Nt-1,NSL::Slice(),NSL::Slice());  // initialize FkFkFk
                                                                                    //
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


} // namespace NSL::FermionMatrix

#endif //NSL_FERMION_MATRIX_HUBBARD_DIAG_TPP
