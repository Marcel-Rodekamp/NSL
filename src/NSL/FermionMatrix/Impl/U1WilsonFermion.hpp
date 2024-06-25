#ifndef NSL_FERMION_MATRIX_WILSON_HPP
#define NSL_FERMION_MATRIX_WILSON_HPP

#include "../fermionMatrix.hpp"
#include "Dirac/gamma.tpp"

namespace NSL::FermionMatrix::U1 {

template<NSL::Concept::isNumber Type>
class Wilson : public NSL::FermionMatrix::FermionMatrix<Type,NSL::Lattice::Square<Type>> {

    public:
    //  No default constructor
    /*  There is no default constructor. */
    Wilson() = delete;

    Wilson(NSL::Lattice::Square<Type> lattice, NSL::Parameter & params):
        // the lattice information is tightly bound to the construction
        // we don't need it here thus just do a default construction which
        // leaves it undefined
        NSL::FermionMatrix::FermionMatrix<Type,NSL::Lattice::Square<Type>>(lattice),
        bareMass_(params["bare mass"].to<Type>()),
        U_(params["device"].to<NSL::Device>(), 
           lattice.sites()
           *params["dim"].to<NSL::size_t>() 
        ),
        dim_(params["dim"].to<NSL::size_t>()),
        gamma_(
            params["device"].to<NSL::Device>(), 
            params["dim"].to<NSL::size_t>()
        )
    {
        NSL::size_t Nt = params["Nt"];
        NSL::size_t Nx = params["Nx"];

        NSL::Tensor<Type> sig3(2*Nx,2*Nx);

        id = NSL::eye<NSL::complex<double>>(2*Nx);
        NSL::Tensor<Type> id_space= NSL::eye<NSL::complex<double>>(Nx);
        sig3(NSL::Slice(0,Nx) ,NSL::Slice(0,Nx)) = id_space;
        sig3(NSL::Slice(Nx,Nx*2) ,NSL::Slice(Nx,Nx*2)) = -1*id_space;

        P0m = 0.5 *(id+sig3);
        P0p = 0.5 *(id-sig3);
    
    }

    void populate(const NSL::Tensor<Type> & phi);

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

    NSL::Tensor<Type> TT(NSL::size_t i, NSL::size_t j); 

    //! Calculate the derivative of M in respect to the input gauge configuration
    //! and apply it to the input psi
    NSL::Tensor<Type> dMdPhi(const NSL::Tensor<Type> & left, const NSL::Tensor<Type> & right);

    //! Calculate the derivative of Mdagger in respect to the input gauge 
    //! configuration and apply it to the input psi
    NSL::Tensor<Type> dMdaggerdPhi(const NSL::Tensor<Type> & left, const NSL::Tensor<Type> & right);

    protected:
    Type bareMass_;

    //! The gauge links exp(i phi ) (N_t, N_x, ..., dim)
    NSL::Tensor<Type> U_;

    NSL::Gamma<Type> gamma_;

    NSL::size_t dim_;
    NSL::Tensor <Type> id;
    NSL::Tensor<Type> P0m;
    NSL::Tensor<Type> P0p;


    
};

} // namespace FermionMatrix

#endif //NSL_FERMION_MATRIX_HUBBARD_EXP_HPP
