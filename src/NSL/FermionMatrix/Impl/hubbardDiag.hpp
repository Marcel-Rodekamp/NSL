#ifndef NSL_FERMION_MATRIX_HUBBARD_DIAG_HPP
#define NSL_FERMION_MATRIX_HUBBARD_DIAG_HPP

/*! \file HubbardDiag.hpp
 *  Class for diagonal discretization of fermion matrix.
 *
 *  The class contains methods which would be used for
 *  the computation of fermionic action
 *  
 **/

#include "../fermionMatrix.hpp"
#include "../../Action/Implementations/hubbard.tpp"

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
class HubbardDiag : public FermionMatrix<Type,LatticeType> {

    public:
        //  No default constructor
        /*  There is no default constructor. */
    HubbardDiag() = delete;

    //!
    /*! 
    *  \param lat  an object of lattice type (Ring, square, etc.).
    *  \param phi  a tensor with dimensions N_t x N_x
    *  \param beta  a floating point number where delta=beta/N_t.
    **/
    
    HubbardDiag(LatticeType & lat, const NSL::size_t Nt, const Type & beta = 1.0, const Type & mu = 0.0 ):
        FermionMatrix<Type,LatticeType>(lat),
        species_(NSL::Hubbard::Species::Particle),
        delta_( beta/Nt ),
        mu_( (beta/Nt)*mu ),
        sgn_( +1 ),
        phi_( lat.device(), Nt, lat.sites() ),
        phiExp_( lat.device(), Nt, lat.sites() ),
        phiExpInv_( lat.device(), Nt, lat.sites() ),
        Fk_( lat.device(), Nt, lat.sites(), lat.sites() ),
        FkFkFk_(lat.device(), Nt, lat.sites(), lat.sites()),
        invAp1F_(lat.device(), lat.sites(), lat.sites()),
        pi_dot_(lat.device(), Nt, lat.sites())
    {}

    HubbardDiag(NSL::Hubbard::Species species, LatticeType & lat, const NSL::size_t Nt, const Type & beta = 1.0, const Type & mu = 0.0 ):
        FermionMatrix<Type,LatticeType>(lat),
        species_(species),
        delta_( beta/Nt ),
        mu_( (beta/Nt)*mu ),
        sgn_( (species == NSL::Hubbard::Particle) ? +1:-1 ),
        phi_( lat.device(), Nt, lat.sites() ),
        phiExp_( lat.device(), Nt, lat.sites() ),
        phiExpInv_( lat.device(), Nt, lat.sites() ),
        Fk_( lat.device(), Nt, lat.sites(), lat.sites() ),
        FkFkFk_(lat.device(), Nt, lat.sites(), lat.sites()),
        invAp1F_(lat.device(), lat.sites(), lat.sites()),
        pi_dot_(lat.device(), Nt, lat.sites())
    {}

    HubbardDiag(NSL::Hubbard::Species species, NSL::Parameter & params):
        HubbardDiag(
            species,
            params["lattice"].to<LatticeType>(), 
            NSL::size_t(params["Nt"]), 
            Type(params["beta"]), 
            Type(params["mu"])
        )
    {}

    HubbardDiag(NSL::Parameter & params):
        HubbardDiag(
            params["lattice"].to<LatticeType>(), 
            NSL::size_t(params["Nt"]), 
            Type(params["beta"]), 
            Type(params["mu"])
        )
    {}

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

    //! Populates the fermion matrix with a new configuration phi
    void populate(const NSL::Tensor<Type> & phi){
        // Reassign phi
        phi_ = phi; 

        // calculate exp(+/- i phi)
        this->phiExp_ = NSL::LinAlg::exp(
            NSL::complex<NSL::RealTypeOf<Type>>(0,sgn_) * phi + sgn_*mu_
        );
        // calculate exp(+/- phi)^{-1} = exp(-/+ i phi)
        this->phiExpInv_ = NSL::LinAlg::exp(
            NSL::complex<NSL::RealTypeOf<Type>>(0,-sgn_) * phi - sgn_*mu_
        );
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

    NSL::Tensor<Type> gradLogDetM() override;

    protected:
    //! species{Particle or Hole} of the fermion matrix
    NSL::Hubbard::Species species_;


    //! delta = beta/N_t
    Type delta_;

    Type mu_;

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
    NSL::Tensor<Type> invAp1F_;
    NSL::Tensor<Type> pi_dot_;

};
} // namespace FermionMatrix

#endif //NSL_FERMION_MATRIX_HUBBARD_DIAG_HPP
