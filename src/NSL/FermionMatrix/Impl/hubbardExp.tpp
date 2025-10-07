#ifndef NSL_FERMION_MATRIX_HUBBARD_EXP_TPP
#define NSL_FERMION_MATRIX_HUBBARD_EXP_TPP

#include "Lattice/lattice.hpp"
#include "device.tpp"
#include "hubbardExp.hpp"
#include "../../Matrix.hpp"
#include "sliceObj.tpp"
#include <tuple>
#include <iterator>
#include <algorithm>

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
        this->Lat.exp_hopping_matrix(sgn_*NSL::LinAlg::conj(delta_)),
        NSL::LinAlg::transpose(psi,-1,-2)
    ).transpose(-1,-2);

    /** We now need to evaluate
      *     (M†ψ)_{tx}  = ψ_tx - exp(-iφ_{tx}^*)      δ_{t+1,i} expKpsi_{ix}
      *     (M†ψ)_{tx}  = ψ_tx - exp(-iφ_{tx}^*)      δ_{t,i-1} expKpsi_{ix}
      *                          |- element-wise * -->|------- shift ------|
      **/
    BexpKpsi.shift(/*shift*/-1,/*dim*/-2,/*boundary*/Type(-1));

    return psi - ( NSL::LinAlg::conj(this->phiExp_) * BexpKpsi);
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
        this->Lat.exp_hopping_matrix(sgn_*delta_),
        (   NSL::LinAlg::shift(this->phiExp_ * NSL::LinAlg::conj(this->phiExp_), +1, -2)
          * NSL::LinAlg::mat_vec(
                this->Lat.exp_hopping_matrix(sgn_*NSL::LinAlg::conj(delta_)),
                NSL::LinAlg::transpose(psi,-1,-2)
            ).transpose(-1,-2)
        ).transpose(-1,-2)
    ).transpose(-1,-2);
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
        this->Lat.exp_hopping_matrix(sgn_*(NSL::LinAlg::conj(delta_)+delta_)),
        (this->phiExp_ * psi ).transpose(-1,-2)
    ).transpose(-1,-2);
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
Type NSL::FermionMatrix::HubbardExp<Type,LatticeType>::logDetM(){
    const int Nt = this->phi_.shape(0);
    const int Nx = this->phi_.shape(1);

    // having a batch dimension requires a bit more work and refactoring of 
    // this algorithm for now we don't implement it here
    assertm( this->phi_.dim() == 2, "NSL::FermionMatrix::HubbardExp::logDetM; phi must be a 2D tensor" );

    NSL::Device device = this->phi_.device();

    NSL::size_t N = NSL::LinAlg::ceil(NSL::LinAlg::log2(static_cast<NSL::RealTypeOf<Type>>(Nt)));
    NSL::size_t full_N = std::pow(2,N);

    NSL::Tensor<Type> prod(device,full_N,Nx,Nx);
    prod = NSL::eye<Type>(device, Nx).expand(full_N,0);
    NSL::Tensor<Type> sausage = NSL::Matrix::Identity<Type>(device,Nx);
    prod(NSL::Slice(0,Nt),NSL::Ellipsis()) = this->Lat.exp_hopping_matrix(sgn_*this->delta_)* NSL::LinAlg::shift(this->phiExp_,-1).expand(Nx).transpose(1,2);

    if (!this->stabilityMethod.compare("SVD")) {
      // use SVD for the stability decomposition

      int primes[50] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
                        67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
			139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
			223, 227, 229};

      NSL::Tensor<Type> U(device,full_N,Nx,Nx); // stores U of M = U.D.V
      NSL::Tensor<Type> D(device,full_N,Nx); // stores D of M = U.D.V
      NSL::Tensor<Type> V(device,full_N,Nx,Nx); // stores V of M = U.D.V

      NSL::Tensor<Type> expKdiag, Uk, Uktemp;
      std::tie(expKdiag, Uk) = this->Lat.eigh_hopping(delta_);  // note, Uk.transpose(Uk) = 1 ::  transpose(Uk).K.Uk = Kdiag, or Uk.expKdiag.transpose(Uk) = expK
      expKdiag.exp();
      if(sgn_ == -1) { // need to reverse order of unitary matrix Uk for holes
        Uktemp = Uk;
	for (int i=0;i<Nx;i++) {
          Uk(NSL::Slice(), i ) = Uktemp(NSL::Slice(), Nx-1-i );
        }	
      }
      
      // initial SVD of B_t
      V(NSL::Slice(0,Nt),NSL::Ellipsis()) = NSL::LinAlg::transpose(Uk) * NSL::LinAlg::shift(this->phiExp_,-1).expand(Nx).transpose(1,2);
      D(NSL::Slice(0,Nt),NSL::Ellipsis()) = expKdiag;
      U(NSL::Slice(0,Nt),NSL::Ellipsis()) = Uk;

      int nnt = Nt;
      int p=0;
      int prime;
      NSL::Tensor<Type> uu(device, Nx, Nx);
      NSL::Tensor<Type> dd(device, Nx);
      NSL::Tensor<Type> vv(device, Nx, Nx);
      NSL::Tensor<Type> udv(device, Nx, Nx);
      NSL::Tensor<Type> vu(device, Nx, Nx);      
      while (nnt>1){
        if (nnt%primes[p] == 0) {
	   prime=primes[p];
           nnt /= prime;
	   for (int tt=0;tt<nnt;tt++) {
	      udv = NSL::LinAlg::diag(D(prime*tt, NSL::Slice()));
	      for (int pr=0;pr<prime-1;pr++){
	         vu = NSL::LinAlg::mat_mul(V(prime*tt+pr, NSL::Slice(), NSL::Slice()) , U(prime*tt+pr+1, NSL::Slice(), NSL::Slice()));
		 udv = NSL::LinAlg::mat_mul( udv, vu );
		 udv = NSL::LinAlg::mat_mul(udv , NSL::LinAlg::diag(D(prime*tt+pr+1, NSL::Slice())));
	      }
	      std::tie( uu, dd, vv ) = NSL::LinAlg::svd(udv(NSL::Slice(),NSL::Slice())); // note that svd returns tuple (U, D, V^\dag)
	      U(tt, NSL::Slice(), NSL::Slice()) = NSL::LinAlg::mat_mul( U(prime*tt, NSL::Slice(), NSL::Slice()) , uu );
	      D(tt, NSL::Slice()) = dd;
	      V(tt, NSL::Slice(), NSL::Slice()) = NSL::LinAlg::mat_mul( vv, V(prime*tt+1, NSL::Slice(), NSL::Slice()) );
           }
	} else {
	  p += 1;
	}
      }

      // I express LogDet(1+U.D.V)=LogDet(U)+LogDet(V)+LogDet(U^T.V^T+D)
      // I then perform a QR decomposition on U^T.V^T+D (which is very stable)
      std::tie( uu, vv ) = NSL::LinAlg::qr(NSL::LinAlg::mat_mul(NSL::LinAlg::adjoint(U(0,NSL::Slice(),NSL::Slice())),NSL::LinAlg::adjoint(V(0,NSL::Slice(),NSL::Slice())))
                                 + NSL::LinAlg::diag(D(0,NSL::Slice()))); // QR decomposition here

      // The final result is LogDet(U)+LogDet(V)+LogDet(Q)+TrLogDiag(R)  (recall that R is upper triangular)
      Type answer = NSL::LinAlg::logdet(U(0,NSL::Slice(),NSL::Slice()))+NSL::LinAlg::logdet(V(0,NSL::Slice(),NSL::Slice()))+NSL::LinAlg::logdet(uu)+(NSL::LinAlg::log(NSL::LinAlg::diag(vv))).sum();
      
      answer = Type (answer.real(), std::remainder(answer.imag(), 6.28318530717959));  // mod the imaginary part back to (-pi,pi)
      
      return answer;

    } else

    if (!this->stabilityMethod.compare("QR")) {
      // use UDT (QR) for the stability decomposition

      int primes[50] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
                        67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
			139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
			223, 227, 229};

      NSL::Tensor<Type> U(device,full_N,Nx,Nx); // stores U of M = U.D.V
      NSL::Tensor<Type> D(device,full_N,Nx); // stores D of M = U.D.V
      NSL::Tensor<Type> V(device,full_N,Nx,Nx); // stores V of M = U.D.V

      NSL::Tensor<Type> expKdiag, Uk, Uktemp;
      std::tie(expKdiag, Uk) = this->Lat.eigh_hopping(delta_);  // note, Uk.transpose(Uk) = 1 ::  transpose(Uk).K.Uk = Kdiag, or Uk.expKdiag.transpose(Uk) = expK
      expKdiag.exp();
      if(sgn_ == -1) { // need to reverse order of unitary matrix Uk for holes
        Uktemp = Uk;
	for (int i=0;i<Nx;i++) {
          Uk(NSL::Slice(), i ) = Uktemp(NSL::Slice(), Nx-1-i );
        }	
      }
      
      // initial SVD of B_t
      V(NSL::Slice(0,Nt),NSL::Ellipsis()) = NSL::LinAlg::transpose(Uk) * NSL::LinAlg::shift(this->phiExp_,-1).expand(Nx).transpose(1,2);
      D(NSL::Slice(0,Nt),NSL::Ellipsis()) = expKdiag;
      U(NSL::Slice(0,Nt),NSL::Ellipsis()) = Uk;

      int nnt = Nt;
      int p=0;
      int prime;
      NSL::Tensor<Type> uu(device, Nx, Nx);
      NSL::Tensor<Type> dd(device, Nx);
      NSL::Tensor<Type> vv(device, Nx, Nx);
      NSL::Tensor<Type> udv(device, Nx, Nx);
      NSL::Tensor<Type> vu(device, Nx, Nx);      
      while (nnt>1){
        if (nnt%primes[p] == 0) {
	   prime=primes[p];
           nnt /= prime;
	   for (int tt=0;tt<nnt;tt++) {
	      udv = NSL::LinAlg::diag(D(prime*tt, NSL::Slice()));
	      for (int pr=0;pr<prime-1;pr++){
	         vu = NSL::LinAlg::mat_mul(V(prime*tt+pr, NSL::Slice(), NSL::Slice()) , U(prime*tt+pr+1, NSL::Slice(), NSL::Slice()));
		 udv = NSL::LinAlg::mat_mul( udv, vu );
		 udv = NSL::LinAlg::mat_mul(udv , NSL::LinAlg::diag(D(prime*tt+pr+1, NSL::Slice())));
	      }
	      std::tie( uu, dd, vv ) = NSL::LinAlg::udt(udv(NSL::Slice(),NSL::Slice())); // note that udt returns tuple (Q, D, (1/D)*R)
	      U(tt, NSL::Slice(), NSL::Slice()) = NSL::LinAlg::mat_mul( U(prime*tt, NSL::Slice(), NSL::Slice()) , uu );
	      D(tt, NSL::Slice()) = dd;
	      V(tt, NSL::Slice(), NSL::Slice()) = NSL::LinAlg::mat_mul( vv, V(prime*tt+1, NSL::Slice(), NSL::Slice()) );
           }
	} else {
	  p += 1;
	}
      }

      // I express LogDet(1+U.D.V)=LogDet(U)+LogDet(V)+LogDet(U^T.V^-1+D)
      // I then perform a QR decomposition on U^T.V^-1+D (which is very stable)
      std::tie( uu, vv ) = NSL::LinAlg::qr( NSL::LinAlg::solve( V(0,NSL::Slice(),NSL::Slice()), NSL::LinAlg::adjoint(U(0,NSL::Slice(),NSL::Slice())), false ) + NSL::LinAlg::diag(D(0,NSL::Slice()))); // QR decomposition here

      // The final result is LogDet(U)+TrLogDiag(V)+LogDet(Q)+TrLogDiag(R)  (recall that R and V are upper triangular)
      Type answer = NSL::LinAlg::logdet(U(0,NSL::Slice(),NSL::Slice()))+NSL::LinAlg::logdet(V(0,NSL::Slice(),NSL::Slice()))+NSL::LinAlg::logdet(uu)+(NSL::LinAlg::log(NSL::LinAlg::diag(vv))).sum();
      
      answer = Type (answer.real(), std::remainder(answer.imag(), 6.28318530717959));  // mod the imaginary part back to (-pi,pi)
      
      return answer;

    } else {

    // Computing F_{Nt-1}.F_{Nt-2}.....F_0 using a recursive tree structure to minimize lost of precision    
    for (NSL::size_t i=0; i<N; i++) {
        prod(NSL::Slice(0, full_N/std::pow(2,i+1)), NSL::Slice(), NSL::Slice()) = NSL::LinAlg::bmm(prod(NSL::Slice(0, full_N/std::pow(2,i), 2), NSL::Slice(), NSL::Slice()), prod(NSL::Slice(1, full_N/std::pow(2,i), 2), NSL::Slice(), NSL::Slice()));
    }

      sausage = prod(0,NSL::Slice(),NSL::Slice());
      return NSL::LinAlg::logdet1plusF(sausage);
    }
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
NSL::Tensor<Type> NSL::FermionMatrix::HubbardExp<Type,LatticeType>::gradLogDetM(){
    //ToDo: implement
    // Define things common to all stability methods
    const int Nt = this->phi_.shape(0);
    const int Nx = this->phi_.shape(1);
    const NSL::Device device = this->phi_.device();

    assertm( this->phi_.dim() == 2, "NSL::FermionMatrix::HubbardExp::logDetM; phi must be a 2D tensor" );

    NSL::complex<NSL::RealTypeOf<Type>> II = NSL::complex<NSL::RealTypeOf<Type>> {0,1.0};
    // Fk(t) = exp(i phi_{x,t-1})^{-1} * exp(-k)
    Fk_ =  NSL::LinAlg::shift(this->phiExpInv_,+1).expand(Nx) * this->Lat.exp_hopping_matrix(-1 * sgn_* this->delta_);

    NSL::size_t N = NSL::LinAlg::ceil(NSL::LinAlg::log2(static_cast<NSL::RealTypeOf<Type>>(Nt)));
    NSL::Tensor<Type> Fkt = NSL::eye<Type>(device,Nx);
    Fkt.expand(Nt+std::pow(2,N)-1, 0);
    Fkt(NSL::Slice(0,Nt), NSL::Ellipsis()) = Fk_;

    // Computing F_{0}^{-1}.F_{1}^{-1}.....F_{Nt-1}^{-1}
    // FkFkFk(t) = Fk(t).Fk(t+1)....Fk(Nt-1)
    // FkFkFk(t=0) gives A^-1 (see eq. 2.32 of Jan-Lukas' notes in hubbardFermionAction.pdf)
    for(int t = 0;  t < N; t++){        
        Fkt(NSL::Slice(std::pow(2, t), Nt+std::pow(2,t+1)-1),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(
            Fkt(NSL::Slice(0, Nt+std::pow(2,t)-1),NSL::Ellipsis()),
            Fkt(NSL::Slice(std::pow(2, t), Nt+std::pow(2,t+1)-1),NSL::Ellipsis())
        );
    } /* I assume that the above multiplication scheme of Petar's and Finn's is correct--T.L. */


    // if we are smart about slicing and copying, we can remove this assignment but I spent too long on other improvements.
    // This is left for the reader
    // FkFkFk0_ = Fkt(NSL::Slice(0,Nt),NSL::Ellipsis());
    // FkFkFk_ = Fkt(NSL::Slice(std::pow(2,N)-1,NSL::None),NSL::Ellipsis());

   
    // Now come the specific stability method parts 
    if (!this->stabilityMethod.compare("DIRECTINVERSE")) {

       NSL::Tensor<Type> invAp1(device, Nx);
       NSL::Tensor<Type> V(device, Nx, Nx);

    // this gives (1+A^-1)^-1  (see eq. 2.31 of Jan-Lukas' notes in hubbardFermionAction.pdf)
       std::tie( invAp1 , V ) = NSL::LinAlg::eig(Fkt(NSL::size_t(std::pow(2,N)-1),NSL::Ellipsis()));  // calculate eigenvalue decomposition of A^{-1}
       invAp1 += 1.;
       invAp1 = 1./invAp1;
       invAp1F_(0,NSL::Ellipsis()) = NSL::LinAlg::mat_mul( V , NSL::LinAlg::solve( V,NSL::LinAlg::diag(invAp1),false ) );  // V * (1/(1+A^{-1})) * V^{-1}

       //std::cout << std::setprecision(25) << invAp1F_(0,NSL::Ellipsis()).real() << std::endl;
       //std::cout << std::setprecision(25) << invAp1F_(0,NSL::Ellipsis()).imag() << std::endl;

    // first do t=Nt-1 case
       pi_dot_(Nt-1,NSL::Slice()) = II * NSL::LinAlg::diag(
           // NSL::LinAlg::mat_mul( FkFkFk_(0,NSL::Ellipsis()), invAp1F_(0,NSL::Ellipsis()) )
           NSL::LinAlg::mat_mul( Fkt(NSL::size_t(std::pow(2,N)-1),NSL::Ellipsis()), invAp1F_(0,NSL::Ellipsis()) )
       );

    // pi_dot_(NSL::Slice(NSL::None,Nt-1),NSL::Ellipsis()) = II * NSL::LinAlg::diagonal(NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(FkFkFk_(NSL::Slice(1,NSL::None),NSL::Ellipsis()),invAp1F_),FkFkFk0_(NSL::Slice(0,Nt-1),NSL::Ellipsis())));
       pi_dot_(NSL::Slice(NSL::None,Nt-1),NSL::Ellipsis()) = II * NSL::LinAlg::diagonal(NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(Fkt(NSL::Slice(std::pow(2,N),NSL::None),NSL::Ellipsis()),invAp1F_),Fkt(NSL::Slice(0,Nt-1),NSL::Ellipsis())));

      return pi_dot_;
      
    }
    else if (!this->stabilityMethod.compare("QR")) {

     int primes[50] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
                        67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
			139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
			223, 227, 229};

      NSL::Tensor<Type> Fkt_U(device,Nt+std::pow(2,N),Nx,Nx); // stores U of M = U.D.V [ = Q.D.(D^{-1}.R) ]
      NSL::Tensor<Type> Fkt_D(device,Nt+std::pow(2,N),Nx);    // stores D of M = U.D.V
      NSL::Tensor<Type> Fkt_V(device,Nt+std::pow(2,N),Nx,Nx); // stores V of M = U.D.V

      NSL::Tensor<Type> expKdiag, Uk, Uktemp;
      std::tie(expKdiag, Uk) = this->Lat.eigh_hopping(delta_);  // note, Uk.transpose(Uk) = 1 ::  Uk.K.transpose(Uk) = Kdiag, or transpose(Uk).expKdiag.Uk = expK

      //if (sgn_ == 1) expKdiag *= -1;
      expKdiag.exp();
      if(sgn_ == 1) { // need to reverse order of unitary matrix Uk for this sign
        Uktemp = Uk;
	for (int i=0;i<Nx;i++) {
          Uk(NSL::Slice(), i ) = Uktemp(NSL::Slice(), Nx-1-i );
        }	
      }

      // initial SVD of Bt^{-1}
      Fkt_U(NSL::Slice(0,Nt),NSL::Ellipsis()) = NSL::LinAlg::shift(this->phiExpInv_,+1).expand(Nx) * Uk;
      Fkt_D(NSL::Slice(0,Nt),NSL::Ellipsis()) = expKdiag;
      Fkt_V(NSL::Slice(0,Nt),NSL::Ellipsis()) = NSL::LinAlg::transpose(Uk);

      /*
      std::cout << "My Fk(0)" << std::endl;
      std::cout << std::setprecision(25) << NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(Fkt_U(0,NSL::Ellipsis()),NSL::LinAlg::diag(Fkt_D(0,NSL::Ellipsis()))), Fkt_V(0,NSL::Ellipsis())).real() << std::endl;
      std::cout << std::setprecision(25) << NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(Fkt_U(0,NSL::Ellipsis()),NSL::LinAlg::diag(Fkt_D(0,NSL::Ellipsis()))), Fkt_V(0,NSL::Ellipsis())).imag() << std::endl;
      std::cout << "My Fk(1)" << std::endl;
      std::cout << std::setprecision(25) << NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(Fkt_U(1,NSL::Ellipsis()),NSL::LinAlg::diag(Fkt_D(1,NSL::Ellipsis()))), Fkt_V(1,NSL::Ellipsis())).real() << std::endl;
      std::cout << std::setprecision(25) << NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(Fkt_U(1,NSL::Ellipsis()),NSL::LinAlg::diag(Fkt_D(1,NSL::Ellipsis()))), Fkt_V(1,NSL::Ellipsis())).imag() << std::endl;
      std::cout << std::endl;
      */

      int nnt = Nt;
      int p=0;
      int prime;
      NSL::Tensor<Type> uu(device, Nx, Nx);
      NSL::Tensor<Type> dd(device, Nx);
      NSL::Tensor<Type> vv(device, Nx, Nx);
      NSL::Tensor<Type> udv(device, Nx, Nx);
      NSL::Tensor<Type> vu(device, Nx, Nx);      
      while (nnt>1){
        if (nnt%primes[p] == 0) {
	   prime=primes[p];
           nnt /= prime;
	   for (int tt=0;tt<nnt;tt++) {
	      udv = NSL::LinAlg::diag(Fkt_D(prime*tt, NSL::Slice()));
	      for (int pr=0;pr<prime-1;pr++){
	         vu = NSL::LinAlg::mat_mul(Fkt_V(prime*tt+pr, NSL::Slice(), NSL::Slice()) , Fkt_U(prime*tt+pr+1, NSL::Slice(), NSL::Slice()));
		 udv = NSL::LinAlg::mat_mul( udv, vu );
		 udv = NSL::LinAlg::mat_mul(udv , NSL::LinAlg::diag(Fkt_D(prime*tt+pr+1, NSL::Slice())));
	      }
	      std::tie( uu, dd, vv ) = NSL::LinAlg::udt(udv(NSL::Slice(),NSL::Slice())); // note that udt returns tuple (Q, D, (1/D)*R)
	      Fkt_U(tt, NSL::Slice(), NSL::Slice()) = NSL::LinAlg::mat_mul( Fkt_U(prime*tt, NSL::Slice(), NSL::Slice()) , uu );
	      Fkt_D(tt, NSL::Slice()) = dd;
	      Fkt_V(tt, NSL::Slice(), NSL::Slice()) = NSL::LinAlg::mat_mul( vv, Fkt_V(prime*tt+1, NSL::Slice(), NSL::Slice()) );
           }
	} else {
	  p += 1;
	}
      }

      NSL::Tensor<Type> Qnew(device, Nx, Nx);
      NSL::Tensor<Type> Dnew(device, Nx);
      NSL::Tensor<Type> Vnew(device, Nx, Nx);

      std::tie( Qnew, Dnew, Vnew ) = NSL::LinAlg::udt( NSL::LinAlg::solve(Fkt_V(0,NSL::Ellipsis()), NSL::LinAlg::adjoint(Fkt_U(0,NSL::Ellipsis())),false ) + NSL::LinAlg::diag(Fkt_D(0,NSL::Ellipsis())));

      invAp1F_U_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew, Fkt_V(0,NSL::Ellipsis())) , NSL::eye<Type>(device,Nx));
      invAp1F_D_(0,NSL::Ellipsis()) = 1./Dnew;
      invAp1F_T_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( Fkt_U(0,NSL::Ellipsis()), Qnew ));

      invAp1F_(0,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(invAp1F_U_(0,NSL::Ellipsis()),NSL::LinAlg::diag(invAp1F_D_(0,NSL::Ellipsis()))),invAp1F_T_(0,NSL::Ellipsis()));

      //invAp1F_(0,NSL::Ellipsis()) = NSL::LinAlg::mat_mul( NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew, Fkt_V(0,NSL::Ellipsis())) , NSL::LinAlg::diag(1./Dnew) ) , NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( Fkt_U(0,NSL::Ellipsis()) , Qnew )) );  // V * (1/(1+A^{-1})) * V^{-1}

      std::cout << std::setprecision(25) << invAp1F_(0,NSL::Ellipsis()).real() << std::endl;
      std::cout << std::setprecision(25) << invAp1F_(0,NSL::Ellipsis()).imag() << std::endl;

      std::cout << std::setprecision(25) << invAp1F_U_(0,NSL::Ellipsis()).real() << std::endl;
      std::cout << std::setprecision(25) << invAp1F_U_(0,NSL::Ellipsis()).imag() << std::endl;
      std::cout << std::setprecision(25) << invAp1F_D_(0,NSL::Ellipsis()).real() << std::endl;
      std::cout << std::setprecision(25) << invAp1F_D_(0,NSL::Ellipsis()).imag() << std::endl;
      std::cout << std::setprecision(25) << invAp1F_T_(0,NSL::Ellipsis()).real() << std::endl;
      std::cout << std::setprecision(25) << invAp1F_T_(0,NSL::Ellipsis()).imag() << std::endl;


    }
    else if (!this->stabilityMethod.compare("SVD")) {

         int primes[50] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
                        67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
			139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
			223, 227, 229};

      NSL::Tensor<Type> Fkt_U(device,Nt+std::pow(2,N),Nx,Nx); // stores U of M = U.D.V [ = Q.D.(D^{-1}.R) ]
      NSL::Tensor<Type> Fkt_D(device,Nt+std::pow(2,N),Nx);    // stores D of M = U.D.V
      NSL::Tensor<Type> Fkt_V(device,Nt+std::pow(2,N),Nx,Nx); // stores V of M = U.D.V

      NSL::Tensor<Type> expKdiag, Uk, Uktemp;
      std::tie(expKdiag, Uk) = this->Lat.eigh_hopping(delta_);  // note, Uk.transpose(Uk) = 1 ::  Uk.K.transpose(Uk) = Kdiag, or transpose(Uk).expKdiag.Uk = expK
      expKdiag.exp();
      if(sgn_ == 1) { // need to reverse order of unitary matrix Uk for this sign
        Uktemp = Uk;
	for (int i=0;i<Nx;i++) {
          Uk(NSL::Slice(), i ) = Uktemp(NSL::Slice(), Nx-1-i );
        }	
      }
      
      // initial SVD of Bt^{-1}
      Fkt_V(NSL::Slice(0,Nt),NSL::Ellipsis()) = NSL::LinAlg::transpose(Uk);
      Fkt_D(NSL::Slice(0,Nt),NSL::Ellipsis()) = expKdiag;
      Fkt_U(NSL::Slice(0,Nt),NSL::Ellipsis()) = NSL::LinAlg::shift(this->phiExpInv_,+1).expand(Nx) * Uk;

      int nnt = Nt;
      int p=0;
      int prime;
      NSL::Tensor<Type> uu(device, Nx, Nx);
      NSL::Tensor<Type> dd(device, Nx);
      NSL::Tensor<Type> vv(device, Nx, Nx);
      NSL::Tensor<Type> udv(device, Nx, Nx);
      NSL::Tensor<Type> vu(device, Nx, Nx);      
      while (nnt>1){
        if (nnt%primes[p] == 0) {
	   prime=primes[p];
           nnt /= prime;
	   for (int tt=0;tt<nnt;tt++) {
	      udv = NSL::LinAlg::diag(Fkt_D(prime*tt, NSL::Slice()));
	      for (int pr=0;pr<prime-1;pr++){
	         vu = NSL::LinAlg::mat_mul(Fkt_V(prime*tt+pr, NSL::Slice(), NSL::Slice()) , Fkt_U(prime*tt+pr+1, NSL::Slice(), NSL::Slice()));
		 udv = NSL::LinAlg::mat_mul( udv, vu );
		 udv = NSL::LinAlg::mat_mul(udv , NSL::LinAlg::diag(Fkt_D(prime*tt+pr+1, NSL::Slice())));
	      }
	      std::tie( uu, dd, vv ) = NSL::LinAlg::svd(udv(NSL::Slice(),NSL::Slice())); // note that udt returns tuple (Q, D, (1/D)*R)
	      Fkt_U(tt, NSL::Slice(), NSL::Slice()) = NSL::LinAlg::mat_mul( Fkt_U(prime*tt, NSL::Slice(), NSL::Slice()) , uu );
	      Fkt_D(tt, NSL::Slice()) = dd;
	      Fkt_V(tt, NSL::Slice(), NSL::Slice()) = NSL::LinAlg::mat_mul( vv, Fkt_V(prime*tt+1, NSL::Slice(), NSL::Slice()) );
           }
	} else {
	  p += 1;
	}
      }

      NSL::Tensor<Type> Qnew(device, Nx, Nx);
      NSL::Tensor<Type> Dnew(device, Nx);
      NSL::Tensor<Type> Vnew(device, Nx, Nx);
      
      std::tie( Qnew , Dnew , Vnew ) = NSL::LinAlg::udt(
      		NSL::LinAlg::mat_mul(NSL::LinAlg::adjoint(Fkt_U(0,NSL::Ellipsis())), NSL::LinAlg::adjoint(Fkt_V(0,NSL::Ellipsis())) )
		+ NSL::LinAlg::diag(Fkt_D(0,NSL::Ellipsis())) );  // Note: I still use QR to stabilize the inverse, even though this is the "SVD" stabilizer routine


      invAp1F_U_(0,NSL::Ellipsis()) = NSL::LinAlg::solve_triangular( Vnew, NSL::LinAlg::adjoint(Fkt_V(0,NSL::Ellipsis())) , false);
      invAp1F_D_(0,NSL::Ellipsis()) = 1./Dnew;
      invAp1F_T_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( Fkt_U(0,NSL::Ellipsis()), Qnew ));

      invAp1F_(0,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(invAp1F_U_(0,NSL::Ellipsis()),NSL::LinAlg::diag(invAp1F_D_(0,NSL::Ellipsis()))),invAp1F_T_(0,NSL::Ellipsis()));

      //invAp1F_(0,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul( NSL::LinAlg::solve_triangular( Vnew, NSL::LinAlg::adjoint(Fkt_V(0,NSL::Ellipsis())) , false),
      //				    NSL::LinAlg::diag(1./Dnew)),NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( Fkt_U(0,NSL::Ellipsis()), Qnew )));

      std::cout << std::setprecision(25) << invAp1F_(0,NSL::Ellipsis()).real() << std::endl;
      std::cout << std::setprecision(25) << invAp1F_(0,NSL::Ellipsis()).imag() << std::endl;

      std::cout << std::setprecision(25) << invAp1F_U_(0,NSL::Ellipsis()).real() << std::endl;
      std::cout << std::setprecision(25) << invAp1F_U_(0,NSL::Ellipsis()).imag() << std::endl;
      std::cout << std::setprecision(25) << invAp1F_D_(0,NSL::Ellipsis()).real() << std::endl;
      std::cout << std::setprecision(25) << invAp1F_D_(0,NSL::Ellipsis()).imag() << std::endl;
      std::cout << std::setprecision(25) << invAp1F_T_(0,NSL::Ellipsis()).real() << std::endl;
      std::cout << std::setprecision(25) << invAp1F_T_(0,NSL::Ellipsis()).imag() << std::endl;


      /*
       * note the line below would be appropriate if we had used SVD to stabilize the inverse

       std::tie( Qnew , Dnew , Vnew ) = NSL::LinAlg::svd(
      		NSL::LinAlg::mat_mul(NSL::LinAlg::adjoint(Fkt_U(0,NSL::Ellipsis())), NSL::LinAlg::adjoint(Fkt_V(0,NSL::Ellipsis())) )
		+ NSL::LinAlg::diag(Fkt_D(0,NSL::Ellipsis())) );

      
      invAp1F_(0,NSL::Ellipsis()) = NSL::LinAlg::mat_mul( NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(NSL::LinAlg::adjoint(Fkt_V(0,NSL::Ellipsis()) ), Vnew) , NSL::LinAlg::diag(1./Dnew) ) , NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( Fkt_U(0,NSL::Ellipsis()), Qnew )) );  // V * (1/(1+A^{-1})) * V^{-1}
      */


    }
    else {
      std::cout << "No valid stability method!!!" << std::endl;
      exit(1);
    }

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


    // the remaining part should not depend on the type of stability method

    // first do t=Nt-1 case
    pi_dot_(Nt-1,NSL::Slice()) = II * NSL::LinAlg::diag(
        // NSL::LinAlg::mat_mul( FkFkFk_(0,NSL::Ellipsis()), invAp1F_(0,NSL::Ellipsis()) )
        NSL::LinAlg::mat_mul( Fkt(NSL::size_t(std::pow(2,N)-1),NSL::Ellipsis()), invAp1F_(0,NSL::Ellipsis()) )
    );

    // pi_dot_(NSL::Slice(NSL::None,Nt-1),NSL::Ellipsis()) = II * NSL::LinAlg::diagonal(NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(FkFkFk_(NSL::Slice(1,NSL::None),NSL::Ellipsis()),invAp1F_),FkFkFk0_(NSL::Slice(0,Nt-1),NSL::Ellipsis())));
    pi_dot_(NSL::Slice(NSL::None,Nt-1),NSL::Ellipsis()) = II * NSL::LinAlg::diagonal(NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(Fkt(NSL::Slice(std::pow(2,N),NSL::None),NSL::Ellipsis()),invAp1F_),Fkt(NSL::Slice(0,Nt-1),NSL::Ellipsis())));

    return pi_dot_;
}

} // namespace FermionMatrix

#endif //NSL_FERMION_MATRIX_HUBBARD_EXP_TPP
