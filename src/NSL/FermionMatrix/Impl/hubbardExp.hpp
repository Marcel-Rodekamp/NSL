#ifndef NSL_FERMION_MATRIX_HUBBARD_EXP_HPP
#define NSL_FERMION_MATRIX_HUBBARD_EXP_HPP

/*! \file HubbardExp.hpp
 *  Class for exponential discretization of fermion matrix.
 *
 *  The class contains methods which would be used for
 *  the computation of fermionic action
 *  
 **/

#include "../fermionMatrix.hpp"
#include "../../Action/Implementations/hubbard.tpp"
#include "Tensor/Factory/like.tpp"

namespace NSL::FermionMatrix {

//! \todo: Remove Hubbard from the name and make it a namespace
/*!
 * The discretization, in index notation, is given by
 *      \f$M_{tx,iy} = δ_{ti} δ_{xy} - [\exp(δK)]_{xy} \exp(i φ_{iy}) B_t δ_{t,i+1}\f$
 *  where
 *   - t and i run over timeslices 0 to \f$N_t-1\f$.
 *   - x and y run over spatial sites
 *   - φ is the auxiliary field
 *   - K is the hopping matrix of the lattice
 *   - Most of the δs are Kronecker deltas; the one multiplying K is β/nt
 *   - B encodes the boundary conditions; \f$B_0 = -1\f$ and all other \f$B=+1\f$.
 *
 * Compare with (12) of \cite Wynen:2018ryx
 *
 **/
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType >
class HubbardExp : public FermionMatrix<Type,LatticeType> {

  
    public:
        //  No default constructor
        /*  There is no default constructor. */
    HubbardExp() = delete;

    //!
    /*! 
    *  \param lat  an object of lattice type (Ring, square, etc.).
    *  \param phi  a tensor with dimensions N_t x N_x
    *  \param beta  a floating point number where delta=beta/N_t.
    **/
    
    HubbardExp(LatticeType & lat, const NSL::size_t Nt, const Type & beta = 1.0, const Type & mu = 0.0 ):
        FermionMatrix<Type,LatticeType>(lat),
	bipartite_(lat.bipartite()),
        species_(NSL::Hubbard::Species::Particle),
        delta_( beta/Nt ),
        mu_( (beta/Nt)*mu ),
        sgn_( +1 ),
        phi_( lat.device(), Nt, lat.sites() ),
        phiExp_( lat.device(), Nt, lat.sites() ),
        phiExpInv_( lat.device(), Nt, lat.sites() ),
        Fk_( lat.device(), Nt, lat.sites(), lat.sites() ),
        FkFkFk_(lat.device(), Nt, lat.sites(), lat.sites()),
        FkFkFk0_(lat.device(), Nt, lat.sites(), lat.sites()),
        invAp1F_(lat.device(), 1, lat.sites(), lat.sites()),
	invAp1F_U_(lat.device(), 1, lat.sites(), lat.sites()),
	invAp1F_D_(lat.device(), 1, lat.sites()),
	invAp1F_T_(lat.device(), 1, lat.sites(), lat.sites()),
        pi_dot_(lat.device(), Nt, lat.sites()),
	expKdiag_(lat.device(), lat.sites()),
	Uk_(lat.device(), lat.sites(), lat.sites()),
	Vk_(lat.device(), lat.sites(), lat.sites()),
	Fkt_U_(lat.device(),Nt,lat.sites(),lat.sites()),
        Fkt_D_(lat.device(),Nt,lat.sites()),
        Fkt_V_(lat.device(),Nt,lat.sites(),lat.sites()),
        fkt_U_(lat.device(),Nt,lat.sites(),lat.sites()),
        fkt_D_(lat.device(),Nt,lat.sites()),
        fkt_V_(lat.device(),Nt,lat.sites(),lat.sites()),
        Fk_U_(lat.device(),lat.sites(),lat.sites()),
        Fk_D_(lat.device(),lat.sites()),   
	Fk_V_(lat.device(),lat.sites(),lat.sites()),
	uu_(lat.device(), lat.sites(), lat.sites()),
	dd_(lat.device(), lat.sites()),
	vv_(lat.device(), lat.sites(), lat.sites()),
	vu_(lat.device(), lat.sites(), lat.sites()),
    vut_(lat.device(), Nt-1, lat.sites(), lat.sites()),
    uut_(lat.device(), Nt-1, lat.sites(), lat.sites()),
	ddt_(lat.device(), Nt-1, lat.sites()),
	vvt_(lat.device(), Nt-1, lat.sites(), lat.sites()),
	udv_(lat.device(), lat.sites(), lat.sites()),
	Qnew_(lat.device(), lat.sites(), lat.sites()),
	Dnew_(lat.device(), lat.sites()),
	Vnew_(lat.device(), lat.sites(), lat.sites()),
    Qnewt_(lat.device(), Nt, lat.sites(), lat.sites()),
	Dnewt_(lat.device(), Nt, lat.sites()),
	Vnewt_(lat.device(), Nt, lat.sites(), lat.sites()),
    Fk_Ut_(lat.device(), Nt, lat.sites(),lat.sites()),
    Fk_Dt_(lat.device(), Nt, lat.sites()),   
	Fk_Vt_(lat.device(), Nt, lat.sites(),lat.sites()),
    invAp1F_Ut_(lat.device(), Nt, lat.sites(), lat.sites()),
	invAp1F_Dt_(lat.device(), Nt, lat.sites()),
	invAp1F_Tt_(lat.device(), Nt, lat.sites(), lat.sites())
    {}

    HubbardExp(NSL::Hubbard::Species species, LatticeType & lat, const NSL::size_t Nt, const Type & beta = 1.0, const Type & mu = 0.0 ):
        FermionMatrix<Type,LatticeType>(lat),
	bipartite_(lat.bipartite()),
        species_(species),
        delta_( beta/Nt ),
        mu_( (beta/Nt)*mu ),
        sgn_( (species == NSL::Hubbard::Particle) ? +1:-1 ),
        phi_( lat.device(), Nt, lat.sites() ),
        phiExp_( lat.device(), Nt, lat.sites() ),
        phiExpInv_( lat.device(), Nt, lat.sites() ),
        Fk_( lat.device(), Nt, lat.sites(), lat.sites() ),
        FkFkFk_(lat.device(), Nt, lat.sites(), lat.sites()),
        FkFkFk0_(lat.device(), Nt, lat.sites(), lat.sites()),
        invAp1F_(lat.device(), 1, lat.sites(), lat.sites()),
	invAp1F_U_(lat.device(), 1, lat.sites(), lat.sites()),
	invAp1F_D_(lat.device(), 1, lat.sites()),
	invAp1F_T_(lat.device(), 1, lat.sites(), lat.sites()),
        pi_dot_(lat.device(), Nt, lat.sites()),
	expKdiag_(lat.device(), lat.sites()),
	Uk_(lat.device(), lat.sites(), lat.sites()),
	Vk_(lat.device(), lat.sites(), lat.sites()),
	Fkt_U_(lat.device(),Nt,lat.sites(),lat.sites()),
        Fkt_D_(lat.device(),Nt,lat.sites()),
        Fkt_V_(lat.device(),Nt,lat.sites(),lat.sites()),
        fkt_U_(lat.device(),Nt,lat.sites(),lat.sites()),
        fkt_D_(lat.device(),Nt,lat.sites()),
        fkt_V_(lat.device(),Nt,lat.sites(),lat.sites()),
        Fk_U_(lat.device(),lat.sites(),lat.sites()),
        Fk_D_(lat.device(),lat.sites()),   
	Fk_V_(lat.device(),lat.sites(),lat.sites()),
	uu_(lat.device(), lat.sites(), lat.sites()),
	dd_(lat.device(), lat.sites()),
	vv_(lat.device(), lat.sites(), lat.sites()),
	vu_(lat.device(), lat.sites(), lat.sites()),
    vut_(lat.device(), Nt-1, lat.sites(), lat.sites()),
    uut_(lat.device(), Nt-1, lat.sites(), lat.sites()),
	ddt_(lat.device(), Nt-1, lat.sites()),
	vvt_(lat.device(), Nt-1, lat.sites(), lat.sites()),
	udv_(lat.device(), lat.sites(), lat.sites()),
	Qnew_(lat.device(), lat.sites(), lat.sites()),
	Dnew_(lat.device(), lat.sites()),
	Vnew_(lat.device(), lat.sites(), lat.sites()),
    Qnewt_(lat.device(), Nt, lat.sites(), lat.sites()),
	Dnewt_(lat.device(), Nt, lat.sites()),
	Vnewt_(lat.device(), Nt, lat.sites(), lat.sites()),
    Fk_Ut_(lat.device(), Nt,lat.sites(),lat.sites()),
    Fk_Dt_(lat.device(), Nt,lat.sites()),   
	Fk_Vt_(lat.device(), Nt,lat.sites(),lat.sites()),
    invAp1F_Ut_(lat.device(), Nt, lat.sites(), lat.sites()),
	invAp1F_Dt_(lat.device(), Nt, lat.sites()),
	invAp1F_Tt_(lat.device(), Nt, lat.sites(), lat.sites())
    {}

    HubbardExp(NSL::Hubbard::Species species, LatticeType & lat, NSL::Parameter & params):
        HubbardExp(species,lat, params["Nt"], params["beta"], params["mu"])
    {}

    HubbardExp(LatticeType & lat, NSL::Parameter & params):
        HubbardExp(lat, params["Nt"], params["beta"], params["mu"])
    {}


    // flag for defining which stability method to use
    std::string stabilityMethod = "QR";  // options are "QR", "DIRECTINVERSE", "SVD"

    //! Populates the fermion matrix with a new configuration phi using species
    /*!
     * For measurements the source vector might by of shape Nt,Nx,Nx (identity matrix
     * on each time slice) This can be prepared by putting an appropriately shaped phi
     * i.e. of shape (Nt,Nx,1). We then need to make space for that calculation, hence
     * expanding the internal tensors to be (tensorShape,1). 
     * If this is desired, set reshape = True. 
     * Otherwise you can use either this function with reshape = False (default)
     * */
    void populate(const NSL::Tensor<Type> & phi, const NSL::Hubbard::Species & species, bool reshape){
        if (reshape and phi_.dim() < phi.dim()) {
            this->expandInternal_();
        } 
        this->populate(phi, species);
    }

    //! Populates the fermion matrix with a new configuration phi
    void populate(const NSL::Tensor<Type> & phi, const NSL::Hubbard::Species & species){
        this->species_ = species;

        if(this->species_ == NSL::Hubbard::Particle){
            this->sgn_ = +1;
        } else {
            this->sgn_ = -1;
        }

        this->populate(phi);
    }

    //! Populates the fermion matrix with a new configuration phi, adding the chemical potential as well
    void populate_w_mu(const NSL::Tensor<Type> & phi, const NSL::Hubbard::Species & species){
        this->species_ = species;

        if(this->species_ == NSL::Hubbard::Particle){
            this->sgn_ = +1;
        } else {
            this->sgn_ = -1;
        }

        this->populate_w_mu(phi);
    }

    //! Populates the fermion matrix with a new configuration phi
    void populate(const NSL::Tensor<Type> & phi){
        // Reassign phi
        phi_ = phi;
	
        // calculate exp(+/- i phi)
        this->phiExp_ = NSL::LinAlg::exp(NSL::complex<NSL::RealTypeOf<Type>>(0,sgn_) * phi);
        // calculate exp(+/- phi)^{-1} = exp(-/+ i phi)
        this->phiExpInv_ = NSL::LinAlg::exp(NSL::complex<NSL::RealTypeOf<Type>>(0,-sgn_) * phi);

    }

    //! Populates the fermion matrix with a new configuration phi, adding the chemical potential as well
    void populate_w_mu(const NSL::Tensor<Type> & phi){
        // Reassign phi
        phi_ = phi;
	
        // calculate exp(+/- i phi)
        this->phiExp_ = NSL::LinAlg::exp(NSL::complex<NSL::RealTypeOf<Type>>(0,sgn_) * phi + sgn_*mu_);
        // calculate exp(+/- phi)^{-1} = exp(-/+ i phi)
        this->phiExpInv_ = NSL::LinAlg::exp(NSL::complex<NSL::RealTypeOf<Type>>(0,-sgn_) * phi - sgn_*mu_);

    }
      

    //! Populates the fermion matrix with a new configuration phi
    /*!
     * For measurements the source vector might by of shape Nt,Nx,Nx (identity matrix
     * on each time slice) This can be prepared by putting an appropriately shaped phi
     * i.e. of shape (Nt,Nx,1). We then need to make space for that calculation, hence
     * expanding the internal tensors to be (tensorShape,1). 
     * If this is desired, set reshape = True. 
     * Otherwise you can use either this function with reshape = False (default)
     * */
    void populate(const NSL::Tensor<Type> & phi, bool reshape){
        if (reshape and phi_.dim() < phi.dim()) {
            this->expandInternal_();
        } 
        this->populate(phi);
    }

    //Declaration of methods methods M, M_dagger, MM_dagger and MdaggerM

    /*!
    *  \param psi a vector with the dimensions N_t x N_x.
    *  \returns M acting on psi, M.psi.
    **/
    NSL::Tensor<Type> M(const NSL::Tensor<Type> & psi) override;

    /*!
    *  \param psi a vector with the dimensions N_t x N_x.
    *  \returns Mdagger acting on psi, Mdagger.psi.
    **/
    NSL::Tensor<Type> Mdagger(const NSL::Tensor<Type> & psi) override;

    /*!
    *  \param psi a vector with the dimensions N_t x N_x.
    *  \returns MMdagger acting on psi, MMdagger.psi.
    **/
    NSL::Tensor<Type> MMdagger(const NSL::Tensor<Type> & psi) override;

    /*!
    *  \param psi a vector with the dimensions N_t x N_x.
    *  \returns MdaggerM acting on psi, MdaggerM.psi.
    **/
    NSL::Tensor<Type> MdaggerM(const NSL::Tensor<Type> & psi) override;

    /*!
    *  \returns log of determinant of M.
    **/
    Type logDetM() override;

    /*!
     * \returns the gradient of log of determinant of M
     **/
    NSL::Tensor<Type> gradLogDetM() override;

    //! Query the current species of the fermion matrix. To change the species please use populate(phi, species).
    NSL::Hubbard::Species species() {
        return species_;
    }

    //! Routine for printing out a matrix (for debugging purposes only!) 
    void printMatrix(const NSL::Tensor<Type> & psi, std::string name, int prec) ;

    //! Bool to state if lattice is bipartite or not
    bool bipartite_;

    //! chemical potential, stored as mu_ = muTilde_ = delta * mu 
    Type mu_;

    protected:
    //! species{Particle or Hole} of the fermion matrix
    NSL::Hubbard::Species species_;


    //! delta = beta/N_t
    Type delta_;

    // Sign of exp( +/- kappa), is assigned in populate
    int sgn_;

    //! The configuration phi (N_t x N_x)
    NSL::Tensor<Type> phi_;
    //! Exponential of phi
    NSL::Tensor<Type> phiExp_;
    //! Inverse Exponential of phi
    NSL::Tensor<Type> phiExpInv_;

    //! Memory used for the implementation of the force
    NSL::Tensor<Type> Fk_;
    NSL::Tensor<Type> FkFkFk_;
    NSL::Tensor<Type> FkFkFk0_;
    NSL::Tensor<Type> invAp1F_;
    NSL::Tensor<Type> pi_dot_;
    NSL::Tensor<Type> invAp1F_U_;
    NSL::Tensor<Type> invAp1F_D_;
    NSL::Tensor<Type> invAp1F_T_;
    NSL::Tensor<Type> expKdiag_, Uk_, Vk_;
    NSL::Tensor<Type> Fkt_U_; //(device,Nt,Nx,Nx); // stores U of M = U.D.V [ = Q.D.(D^{-1}.R) ]
    NSL::Tensor<Type> Fkt_D_; //(device,Nt,Nx);    // stores D of M = U.D.V
    NSL::Tensor<Type> Fkt_V_; //(device,Nt,Nx,Nx); // stores V of M = U.D.V
    NSL::Tensor<Type> fkt_U_; //(device,Nt,Nx,Nx); // stores U of M = U.D.V [ = Q.D.(D^{-1}.R) ]
    NSL::Tensor<Type> fkt_D_; //(device,Nt,Nx);    // stores D of M = U.D.V
    NSL::Tensor<Type> fkt_V_; //(device,Nt,Nx,Nx); // stores V of M = U.D.V
    NSL::Tensor<Type> Fk_U_; //(device,Nx,Nx);
    NSL::Tensor<Type> Fk_D_; //(device,Nx);   
    NSL::Tensor<Type> Fk_V_; //(device,Nx,Nx);
    NSL::Tensor<Type> uu_; //(device, Nx, Nx);
    NSL::Tensor<Type> dd_; //(device, Nx);
    NSL::Tensor<Type> vv_; //(device, Nx, Nx);
    NSL::Tensor<Type> vu_; //(device, Nx, Nx);
    NSL::Tensor<Type> vut_; //(device, Nt-1, Nx, Nx);
    NSL::Tensor<Type> uut_; //(device, Nx, Nx);
    NSL::Tensor<Type> ddt_; //(device, Nx);
    NSL::Tensor<Type> vvt_; //(device, Nx, Nx);
    NSL::Tensor<Type> udv_; //(device, Nx, Nx);
    NSL::Tensor<Type> Qnew_; //(device, Nx, Nx);
    NSL::Tensor<Type> Dnew_; //(device, Nx);
    NSL::Tensor<Type> Vnew_; //(device, Nx, Nx);
    NSL::Tensor<Type> Qnewt_; 
	NSL::Tensor<Type> Dnewt_;
	NSL::Tensor<Type> Vnewt_; 
    NSL::Tensor<Type> Fk_Ut_;
    NSL::Tensor<Type> Fk_Dt_;   
	NSL::Tensor<Type> Fk_Vt_;
    NSL::Tensor<Type> invAp1F_Ut_;
	NSL::Tensor<Type> invAp1F_Dt_;
	NSL::Tensor<Type> invAp1F_Tt_;

    //!  prime numbers used in the recursive tree calculation of loddet
    int primes_[50] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
                        67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
			139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
			223, 227, 229};

    /*!
     * F_(psi) returns a vector the same shape as \f$\psi\f$ that is given by
     *
     * \f$ \texttt{F_(psi)}_{tx} = [\exp(δK)]_{xy} \exp(i φ_{iy}) B_t δ_{t,i+1} \psi_{yi}\f$
     *
     * which is the off-diagonal piece of \f$M\f$ itself applied to an appropriate vector.
     *
     **/
    NSL::Tensor<Type> F_(const NSL::Tensor<Type> & psi);

    void expandInternal_(){
        phi_.expand(1);
        phiExp_.expand(1);
        phiExpInv_.expand(1);
        Fk_.expand(1);
        FkFkFk_.expand(1);
        FkFkFk0_.expand(1);
        invAp1F_.expand(1);
        pi_dot_.expand(1);
	invAp1F_U_.expand(1);
	invAp1F_D_.expand(1);
	invAp1F_T_.expand(1);
    }
};
} // namespace FermionMatrix

#endif //NSL_FERMION_MATRIX_HUBBARD_EXP_HPP
