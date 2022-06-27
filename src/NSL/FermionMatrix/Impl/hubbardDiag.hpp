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
    
    HubbardDiag(LatticeType & lat,  const NSL::Tensor<Type> &phi, const Type & beta = 1.0 ):
        FermionMatrix<Type,LatticeType>(lat),
        phi_(phi),
        delta_(beta/phi.shape(0)),
        phiExp_(NSL::LinAlg::exp(phi*NSL::complex<typename NSL::RT_extractor<Type>::value_type>(0,1)))
    {}

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

    protected:
    //! The configuration phi (N_t x N_x)
    NSL::Tensor<Type> phi_;
    //! Exponential of phi
    NSL::Tensor<Type> phiExp_;
    //! delta = beta/N_t
    Type delta_;

};
} // namespace FermionMatrix

#endif //NSL_FERMION_MATRIX_HUBBARD_DIAG_HPP