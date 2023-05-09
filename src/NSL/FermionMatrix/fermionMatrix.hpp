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
#include "../Matrix.hpp"
#include <type_traits>
#include <ATen/cuda/CUDAGraph.h>

namespace NSL::FermionMatrix{

//definition of class FermionMatrix

/*! A base class for exponential dizcreatization of the fermion
*   matrix
**/

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType >
class FermionMatrix {

    public:

    //Declaration of methods methods M, M_dagger, MM_dagger and MdaggerM

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

    /*!
     * \returns the gradient of log of determinant of M
     **/
    NSL::Tensor<Type> gradLogDetM();

    //! This is an optimized version of the gradient of log of determinant of M
    NSL::Tensor<Type> gradLogDetM(std::vector<at::cuda::CUDAGraph> & graphs);

    // constructors
    /*  There is no default constructor. */
    FermionMatrix() = delete;
    FermionMatrix(FermionMatrix<Type,LatticeType> &) = delete;
    FermionMatrix(FermionMatrix<Type,LatticeType> &&) = delete;
    /*! 
    *  \param lat  an object of Lattice type (Ring, square, etc.).
    **/
    FermionMatrix(LatticeType & lat):
        Lat(lat)
    {}

    // Construction
    // Returns a tensor with dimensions (nt, nx, nt, nx) where
    //  - the first  two are the row,
    //  - the latter two are the column.
    // Of course, you should avoid constructing the dense matrix while in production unless absolutely necessary!
    // TODO: I'm not sure this should actually be a method of the root-level fermionMatrix class, since it assumes that a configuration is of shape (nt, nt) and no other internal indices.
    NSL::Tensor<Type> M_dense(NSL::size_t nt){
        // We construct a dense operator generically by applying the mandatory M routine nt * nx times.
        // We leverage the fact that
        //      M_{tx,iy} = M_{tx,jz} * δ_{ji} δ_{zy}
        // On the left-hand side we have a dense matrix;
        // we get the right-hand side by mat-vec on nt * nx vectors labeled by iy.

        // First we construct the identity matrix.
        auto nx = this->Lat.sites();
        NSL::Tensor<Type> identity(nt, nx, nt, nx);
        for(int t = 0; t < nt; t++){
            identity(t,NSL::Slice(), t, NSL::Slice()) = NSL::Matrix::Identity<Type>(nx);
        }

        // Then we apply M to each column.
        NSL::Tensor<Type> M(nt, nx, nt, nx);
        for(int i = 0; i < nt; i++){
            for(int y = 0; y < nx; y++){
                M(NSL::Slice(), NSL::Slice(), i, y) = this->M(identity(NSL::Slice(), NSL::Slice(), i, y));
            }
        }

        return M;
    }

    protected:
    //! An object of Lattice type (Ring, square, etc.).
    LatticeType & Lat;
};
}

#endif
