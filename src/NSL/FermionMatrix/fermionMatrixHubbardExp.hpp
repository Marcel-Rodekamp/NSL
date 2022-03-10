#ifndef NANOSYSTEMLIBRARY_FERMIONMATRIXHUBBARDEXP_HPP
#define NANOSYSTEMLIBRARY_FERMIONMATRIXHUBBARDEXP_HPP

/*! \file fermionMatrixHubbardExp.hpp
 *  Class for exponential discretization of fermion matrix.
 *
 *  The class contains methods which would be used for
 *  the computation of fermionic action
 *    
 **/

#include "LinAlg/mat_vec.hpp"
#include "fermionMatrixBase.hpp"
#include "LinAlg/mat_conj.hpp"
#include "LinAlg/mat_mul.hpp"
#include "LinAlg/mat_exp.hpp"
#include "LinAlg/mat_trans.hpp"
#include "LinAlg/mat_inv.hpp"
#include "LinAlg/det.hpp"
#include "LinAlg/exp.hpp"
#include "LinAlg/matrix.hpp"


namespace NSL::FermionMatrix {
template<typename Type>

//! \todo: Remove FermionMatrix from the name
//! \todo: Remove Hubbard from the name and make it a namespace
class FermionMatrixHubbardExp : public FermionMatrixBase<Type> {

    public:
        //  No default constructor
        /*  There is no default constructor. */
    FermionMatrixHubbardExp() = delete;

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
    
    FermionMatrixHubbardExp(NSL::Lattice::SpatialLattice<typename NSL::RT_extractor<Type>::value_type> *lat,  const NSL::TimeTensor<Type> &phi, double beta = 1.0 ):
        FermionMatrixBase<Type>(lat),
        phi_(phi),
        delta_(beta/phi.shape(0)),
        phiExp_(NSL::LinAlg::exp(phi*NSL::complex<typename NSL::RT_extractor<Type>::value_type>(0,1)))
        

    {}

    //Declaration of methods methods M, M_dagger, MM_dagger and M

    /*!
    *  \param psi a vector with the dimensions N_t x N_x.
    *  \returns M acting on psi, M.psi.
    **/
    NSL::TimeTensor<Type> M(const NSL::TimeTensor<Type> & psi) override;

    /*!
    *  \param psi a vector with the dimensions N_t x N_x.
    *  \returns Mdagger acting on psi, Mdagger.psi.
    **/
    NSL::TimeTensor<Type> Mdagger(const NSL::TimeTensor<Type> & psi) override;

    /*!
    *  \param psi a vector with the dimensions N_t x N_x.
    *  \returns MMdagger acting on psi, MMdagger.psi.
    **/
    NSL::TimeTensor<Type> MMdagger(const NSL::TimeTensor<Type> & psi) override;

    /*!
    *  \param psi a vector with the dimensions N_t x N_x.
    *  \returns MdaggerM acting on psi, MdaggerM.psi.
    **/
    NSL::TimeTensor<Type> MdaggerM(const NSL::TimeTensor<Type> & psi) override;

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
    NSL::RT_extractor<Type>::value_type delta_;

    private:
    NSL::TimeTensor<Type> F_(const NSL::TimeTensor<Type> & psi);

};
} // namespace FermionMatrix

#endif //NANOSYSTEMLIBRARY_FERMIONMATRIXHUBBARDEXP_HPP
