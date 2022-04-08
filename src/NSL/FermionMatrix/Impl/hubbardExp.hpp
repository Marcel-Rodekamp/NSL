#ifndef NSL_FERMION_MATRIX_HUBBARD_EXP_HPP
#define NSL_FERMION_MATRIX_HUBBARD_EXP_HPP

/*! \file fermionMatrixHubbardExp.hpp
 *  Class for exponential discretization of fermion matrix.
 *
 *  The class contains methods which would be used for
 *  the computation of fermionic action
 *    
 **/

#include "../fermionMatrix.hpp"

namespace NSL::FermionMatrix {

//! \todo: Remove FermionMatrix from the name
//! \todo: Remove Hubbard from the name and make it a namespace
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType >
class HubbardExp : public FermionMatrix<Type,LatticeType> {

    public:
        //  No default constructor
        /*  There is no default constructor. */
    HubbardExp() = delete;

    //!
    /*!
     *
     * \todo: SpatialLattice accepts only real type template arguments
     *   
    *! 
    *  \param lat  an object of lattice type (Ring, square, etc.).
    *  \param phi  a tensor with dimensions N_t x N_x
    *  \param beta  a floating point number where delta=beta/N_t.
    **/
    
    HubbardExp(LatticeType & lat,  const NSL::Tensor<Type> &phi, const Type & beta = 1.0 ):
        FermionMatrix<Type,LatticeType>(lat),
        phi_(phi),
        delta_(beta/phi.shape(0)),
        phiExp_(NSL::LinAlg::exp(phi*NSL::complex<typename NSL::RT_extractor<Type>::value_type>(0,1)))
    {}

    //Declaration of methods methods M, M_dagger, MM_dagger and M

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

    private:
    NSL::Tensor<Type> F_(const NSL::Tensor<Type> & psi);

};
} // namespace FermionMatrix

#endif //NSL_FERMION_MATRIX_HUBBARD_EXP_HPP
