#ifndef NSL_SOLVER_HPP
#define NSL_SOLVER_HPP

#include "../../concepts.hpp"
#include "../../Lattice/lattice.hpp"
#include "../../FermionMatrix.hpp"

#include "applicationType.tpp"

#include <functional>
#include <stdexcept>

namespace NSL::LinAlg {

//! Base Class for solvers
/*!
 *  This class provides the interface for algorithms that solve equations
 *      \f[ 
 *          M x = b
 *      \f]
 *  for x, where M is a (fermion-)matrix and x,b are vectors.
 *  
 *  There are two versions of this interface, one specialized for fermion
 *  matrices defined via `NSL::FermionMatrix::FermionMatrix` and one for a
 *  general dense matrix
 *
 *  The actual algorithms can be found in `src/NSL/LinAlg/Solver/Impl`.
 *
 * */
// These templates are automatically deduced by the compiler!
template<
    // The number type used for the solve, e.g. float,double, NSL::complex<float> ...
    NSL::Concept::isNumber Type
>
class Solver{
    public:
        
        Solver() = delete;
        Solver(const Solver<Type> & ) = default;
        Solver(Solver<Type> &&) = default;

        //! Constructor
        /*! 
         * \param M
         *        Matrix times vector application for which the equation 
         *          \f[ M x = b \f]
         *        is sovled for x.
         * */
        Solver(std::function<NSL::Tensor<Type>(const NSL::Tensor<Type> &)> M) : M_(M){}

        //! Constructor
        /*! 
         * \param M
         *        derived object of `NSL::FermionMatrix::FermionMatrix`, 
         *        a fermion matrix for which the equation 
         *          \f[ M x = b \f]
         *        is sovled for x.
         * \param function_ptr
         *        this specifies which application of the 
         *        fermion matrix `M`,`Mdagger`,`MdaggerM`,`MMdagger` shall
         *        solved. You can use explesstions like
         *          * &NSL::FermionMatrix::FermionMatrix<Type,NSL::Lattice::SpatialLattice<Type>>::M (default if not provided)
         *          * &NSL::FermionMatrix::FermionMatrix<Type,NSL::Lattice::SpatialLattice<Type>>::Mdagger 
         *          * &NSL::FermionMatrix::FermionMatrix<Type,NSL::Lattice::SpatialLattice<Type>>::MdaggerM 
         *          * &NSL::FermionMatrix::FermionMatrix<Type,NSL::Lattice::SpatialLattice<Type>>::MMdagger 
         *
         * \param `FermionMatrix<TypeHelper,LatticeHelper>`(Template)
         *                   This template defines the type of Fermion Matrix, as any fermion 
         *                   matrix derived from NS::FermionMatrix::FermionMatrix requires two
         *                   template arguments (Type,Lattice) we have to give a template as template argument.
         *                   Hereby the usage is: FermionMatrix<Type,LatticeType>
         *                   defining the final type used throughout the classes.
         * \param `LatticeType`(Template)
         *                   This defines the LatticeType used for the FermionMatrix template template argument.
         *                   It is checked that it derives from NSL::Lattice::SpatialLattice as to
         *                   ensure that the required interface is given.
         * */
        template<
            template<typename TypeHelper, typename LatticeHelper> class FermionMatrix,
            NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType
        >
            // Check that the given FermionMatrix<Type,LatticeType> is
            // deriving from NSL::FermionMatrix::FermionMatrix<Type,LatticeType> 
            // to ensure that the required interface is given.
            requires( NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>,FermionMatrix<Type,LatticeType>> )
        Solver(FermionMatrix<Type,LatticeType> & M, 
               NSL::FermionMatrix::MatrixCombination matrixCombination = NSL::FermionMatrix::M
        ) {
            switch(matrixCombination){
                case NSL::FermionMatrix::M        : M_ = ( [&M](const NSL::Tensor<Type> & psi){ return M.M(psi);} ); break;
                case NSL::FermionMatrix::Mdagger  : M_ = ( [&M](const NSL::Tensor<Type> & psi){ return M.Mdagger(psi);} ); break;
                case NSL::FermionMatrix::MdaggerM : M_ = ( [&M](const NSL::Tensor<Type> & psi){ return M.MdaggerM(psi);} ); break;
                case NSL::FermionMatrix::MMdagger : M_ = ( [&M](const NSL::Tensor<Type> & psi){ return M.MMdagger(psi);} ); break;
                default: throw std::runtime_error("NSL::Solver: Could not identify fermion matrix combination pass either NSL:FermionMatrix::(M,Mdagger,MdaggerM or MMdagger)!");
            }
        }
        //Solver(const FermionMatrix<Type,LatticeType> & M, 
        //       NSL::Tensor<Type> (FermionMatrix<Type,LatticeType>::* function_ptr)(const NSL::Tensor<Type> &)) : 
        //    M_(std::bind_front( function_ptr,&M ))
        //{}

        //! Apply Solver
        /*!
         *  \param b, NSL::Tensor, RHS of the equation to be solved
         *
         * This operator is overloaded by the various implementations and 
         * performs the solve of 
         * \f[
         *      M x = b
         * \f]
         * It returns an NSL::Tensor being the (approximate) solution
         * \f[
         *      x = M^{-1} b
         * \f]
         * for the stored fermion matrix M.
         * */
        NSL::Tensor<Type> operator()(const NSL::Tensor<Type> & b);

    protected:
        std::function<NSL::Tensor<Type>(const NSL::Tensor<Type> &)> M_;
};

} // namespace NSL::LinAlg

#endif // NSL_SOLVER_HPP
