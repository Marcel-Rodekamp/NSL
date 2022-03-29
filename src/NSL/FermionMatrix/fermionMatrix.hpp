#ifndef NSL_FERMIONMATRIX_HPP
#define NSL_FERMIONMATRIX_HPP

/*! \file fermionMatrix.hpp
 *  Class for exponential discretization of fermion matrix.
 *
 *  The class contains methods which would be used for
 *  the computation of fermionic action
 *    
 **/

#include "../Lattice.hpp"
#include "../Tensor.hpp"

namespace NSL::FermionMatrix{

//definition of class FermionMatrix

/*! A base class for exponential dizcreatization of the fermion
*   matrix
**/

template<NSL::Concept::isNumber Type>
class FermionMatrix {

    public:

    //Declaration of methods methods M, M_dagger, MM_dagger and M

    /*!
    *  \param psi a vector with the dimensions N_t x N_x.
    *  \returns M acting on psi, M.psi.
    **/
    virtual NSL::Tensor<Type> M(const NSL::Tensor<Type> & psi) = 0;

    /*!
    *  \param psi a vector with the dimensions N_t x N_x.
    *  \returns Mdagger acting on psi, Mdagger.psi.
    **/
    virtual NSL::Tensor<Type> Mdagger(const NSL::Tensor<Type> & psi) = 0;

    /*!
    *  \param psi a vector with the dimensions N_t x N_x.
    *  \returns MMdagger acting on psi, MMdagger.psi.
    **/
    virtual NSL::Tensor<Type> MMdagger(const NSL::Tensor<Type> & psi) = 0;

    /*!
    *  \param psi a vector with the dimensions N_t x N_x.
    *  \returns MdaggerM acting on psi, MdaggerM.psi.
    **/
    virtual NSL::Tensor<Type> MdaggerM(const NSL::Tensor<Type> & psi) = 0;

    /*!
    *  \returns log of determinant of M.
    **/
    virtual Type logDetM() = 0;

    // constructors
    /*  There is no default constructor. */
    FermionMatrix() = delete;
    FermionMatrix(FermionMatrix<Type> &) = delete;
    FermionMatrix(FermionMatrix<Type> &&) = delete;
    /*! 
    *  \param lat  an object of Lattice type (Ring, square, etc.).
    **/
    FermionMatrix(NSL::Lattice::SpatialLattice<typename NSL::RT_extractor<Type>::value_type> * lat):
        Lat(lat)
    {}

    protected:
    //! An object of Lattice type (Ring, square, etc.).
    NSL::Lattice::SpatialLattice<typename NSL::RT_extractor<Type>::value_type>* Lat;
};
}

#endif