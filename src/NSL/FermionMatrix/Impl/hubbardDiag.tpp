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
        this->Lat.hopping_matrix(delta_),
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
    
    return psi - NSL::LinAlg::mat_vec(this->Lat.hopping_matrix(NSL::LinAlg::conj(delta_)), NSL::LinAlg::transpose(psi)).transpose() - Bexpconjphipsi;
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
  out = this->M(psi) + this->Mdagger(psi) - psi + NSL::LinAlg::mat_vec(this->Lat.hopping_matrix((delta_)), NSL::LinAlg::mat_vec(this->Lat.hopping_matrix((NSL::LinAlg::conj(delta_))),NSL::LinAlg::transpose(psi))).transpose();
  KexpconjphiBpsi = psi;
  KexpconjphiBpsi(0,NSL::Slice()) *= -1;
  out += NSL::LinAlg::mat_vec(this->Lat.hopping_matrix(delta_), (NSL::LinAlg::conj(this->phiExp_) * KexpconjphiBpsi.shift(-1)).transpose()).transpose();
 
  BexpconjdelKpsi = (this->phiExp_) * NSL::LinAlg::mat_vec(this->Lat.hopping_matrix(NSL::LinAlg::conj(delta_)), NSL::LinAlg::transpose(psi)).transpose();
  BexpconjdelKpsi.shift(1);
  BexpconjdelKpsi(0,NSL::Slice()) *= -1;
  out += BexpconjdelKpsi + (((this->phiExp_) * NSL::LinAlg::conj(this->phiExp_)).shift(1)) * psi;
  return (out);
  
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
  out = this->M(psi) + this->Mdagger(psi) - psi + NSL::LinAlg::mat_vec(this->Lat.hopping_matrix((delta_)), NSL::LinAlg::mat_vec(this->Lat.hopping_matrix((NSL::LinAlg::conj(delta_))),NSL::LinAlg::transpose(psi))).transpose();
  KexpconjphiBpsi = psi;
  KexpconjphiBpsi(0,NSL::Slice()) *= -1;
  out += NSL::LinAlg::conj(this->phiExp_) * NSL::LinAlg::mat_vec(this->Lat.hopping_matrix(delta_), KexpconjphiBpsi.shift(-1).transpose()).transpose();
  
  BexpconjdelKpsi =   NSL::LinAlg::mat_vec(this->Lat.hopping_matrix(NSL::LinAlg::conj(delta_)), ((this->phiExp_) * psi).transpose()).transpose();
  BexpconjdelKpsi.shift(1);
  BexpconjdelKpsi(0,NSL::Slice()) *= -1;
  out += BexpconjdelKpsi + ((this->phiExp_) * NSL::LinAlg::conj(this->phiExp_)) * psi;

  return (out);

}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
Type NSL::FermionMatrix::HubbardDiag<Type,LatticeType>::logDetM(){
    const int Nt = this->phi_.shape(0);
    const int Nx = this->phi_.shape(1); 
    
    NSL::Device dev = phi_.device();

    NSL::Tensor<Type> prod(dev,Nt,Nx,Nx);
    NSL::Tensor<Type> sausage = NSL::Matrix::Identity<Type>(dev,Nx), id(dev,Nx,Nx), fk(dev,Nx,Nx);
    id = sausage.to(dev);
    //id = sausage;
    //id.to(NSL::GPU());
    Type sum, I = {0,1};
    
    //F^{-1} in expanded form    
    prod = (NSL::LinAlg::shift(NSL::LinAlg::exp(this->phi_*(-1*I)),-1).expand(Nx).transpose(1,2));

    for(int t = 0;  t < Nt ; t++){

        //mat-vec of F^{-1} and K
        fk = prod(t,NSL::Slice(),NSL::Slice())* (id - this->Lat.hopping_matrix(delta_));
        
        sausage.mat_mul(fk); 
    }
    //sum over all the elements of phi 
    sum = (this->phi_).sum(); 

    
    return ((sum*I) + NSL::LinAlg::logdet(NSL::Matrix::Identity<Type>(dev,Nx) + sausage));
} 
}

#endif //NSL_FERMION_MATRIX_HUBBARD_DIAG_TPP
