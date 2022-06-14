#ifndef NSL_CG_HPP
#define NSL_CG_HPP

#include "../Solver.hpp" 
#include "complex.hpp"
#include "types.hpp"

namespace NSL::LinAlg {

//! Conjugate Gradient, solving \f$M x = b \f$
template<NSL::Concept::isNumber Type>
class CG: public NSL::LinAlg::Solver<Type> {
    public:
        
        //! Constructor
        /*! 
         * \param M
         *        Matrix times vector application for which the equation 
         *          \f[ M x = b \f]
         *        is sovled for x.
         * \param eps
         *        Error at which the CG is stopped as 
         *          \f[ \vert\vert Mx_i - b\vert\vert^2 < \texttt{eps} \f]
         * \param maxIter
         *        In case the CG doesn't converge this is a fall back to 
         *        exit. If the iteration count exeeds this number a runtime
         *        error is raised.
         *
         * This Solver implementation uses the conjugate gradient (CG) algorithm.
         * */
        CG(std::function<NSL::Tensor<Type>(const NSL::Tensor<Type> &)> M,
               const typename NSL::RT_extractor<Type>::type eps = 1e-6, const NSL::size_t maxIter = 10000) : 
            NSL::LinAlg::Solver<Type>(M),
            errSq_(eps*eps),
            maxIter_(maxIter),
            x_(),
            t_(),
            r_(),
            p_()
        {}

        //! Constructor
        /*! 
         * \param M 
         *        derived object of `NSL::FermionMatrix::FermionMatrix`, 
         *        a fermion matrix for which the equation 
         *          \f[ M x = b \f]
         *        is sovled for x.
         * \param eps
         *        Error at which the CG is stopped as 
         *          \f[ \vert\vert Mx_i - b\vert\vert^2 < \texttt{eps} \f]
         * \param maxIter
         *        In case the CG doesn't converge this is a fall back to 
         *        exit. If the iteration count exeeds this number a runtime
         *        error is raised.
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
         *
         * This default case uses the application of the fermion matrix
         * ```
         *      M = NSL::FermionMatrix::FermionMatrixImpl.M
         * ```
         *
         * This Solver implementation uses the conjugate gradient (CG) algorithm.
         *
         * */
        template<
            template<typename TypeHelper, typename LatticeHelper> class FermionMatrix,
            NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType
        >
            // Check that the given FermionMatrix<Type,LatticeType> is
            // deriving from NSL::FermionMatrix::FermionMatrix<Type,LatticeType> 
            // to ensure that the required interface is given.
            requires( NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>,FermionMatrix<Type,LatticeType>> )
        CG(FermionMatrix<Type,LatticeType> & M,
               const typename NSL::RT_extractor<Type>::type eps = 1e-6, const NSL::size_t maxIter = 10000) : 
            NSL::LinAlg::Solver<Type>(M, NSL::FermionMatrix::M),
            errSq_(eps*eps),
            maxIter_(maxIter),
            x_(),
            t_(),
            r_(),
            p_()
        {}

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
         * \param eps
         *        Error at which the CG is stopped as 
         *          \f[ \vert\vert Mx_i - b\vert\vert^2 < \texttt{eps} \f]
         * \param maxIter
         *        In case the CG doesn't converge this is a fall back to 
         *        exit. If the iteration count exeeds this number a runtime
         *        error is raised.
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
         *
         * This default case uses the application of ther fermion matrix
         * ```
         *      M = *function_ptr 
         * ```
         *
         * This Solver implementation uses the conjugate gradient (CG) algorithm.
         * */
        template<
            template<typename TypeHelper, typename LatticeHelper> class FermionMatrix,
            NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType
        >
            // Check that the given FermionMatrix<Type,LatticeType> is
            // deriving from NSL::FermionMatrix::FermionMatrix<Type,LatticeType> 
            // to ensure that the required interface is given.
            requires( NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>,FermionMatrix<Type,LatticeType>> )
        CG(FermionMatrix<Type,LatticeType> & M, 
               NSL::FermionMatrix::MatrixCombination matrixCombination,
               const typename NSL::RT_extractor<Type>::type eps = 1e-6, const NSL::size_t maxIter = 10000) : 
            NSL::LinAlg::Solver<Type>(M,matrixCombination),
            errSq_(eps*eps),
            maxIter_(maxIter),
            x_(),
            t_(),
            r_(),
            p_()
        {}

        //! Apply CG
        /*!
         *  \param b, NSL::Tensor, RHS of the equation to be solved
         *
         * This operator performs the solve of 
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

    private:

        // precision at which the algorithm is stopped
        const typename NSL::RT_extractor<Type>::type errSq_;
        // maximum of iterations as fall back in case we don't converge
        const NSL::size_t maxIter_;

        // vector to store intermediate solution
        NSL::Tensor<Type> x_;
        // vector to store the result of M @ v 
        NSL::Tensor<Type> t_;
        // residual vector
        NSL::Tensor<Type> r_;
        // gradient vector
        NSL::Tensor<Type> p_;
}; // class CG
        
} //namespace NSL::LinAlg

#endif // NSL_CG_HPP
