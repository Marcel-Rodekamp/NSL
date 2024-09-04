#ifndef NSL_CG_PREC_HPP
#define NSL_CG_PREC_HPP

#include "../Solver.hpp" 
#include "CG.hpp"
#include "complex.hpp"
#include "types.hpp"

namespace NSL::LinAlg {

//! Conjugate Gradient, solving \f$M x = b \f$
template<NSL::Concept::isNumber Type>
class CGpreconditioned: public NSL::LinAlg::Solver<Type> {
    public:
        
        //! Constructor
        /*! 
         * \param M
         *        Matrix times vector application for which the equation 
         *          \f[ M x = b \f]
         *        is sovled for x.
         * \param Mprec
         *        Preconditioner matrix
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
        CGpreconditioned(
                std::function<NSL::Tensor<Type>(const NSL::Tensor<Type> &)> M,
                std::function<NSL::Tensor<Type>(const NSL::Tensor<Type> &)> Mprec,
                const NSL::RealTypeOf<Type> eps = 1e-12, const NSL::size_t maxIter = 10000) : 
            NSL::LinAlg::Solver<Type>(M),
            innerCG_(Mprec, eps, maxIter),
            errSq_(eps*eps),
            maxIter_(maxIter),
            x_(),
            t_(),
            r_(),
            z_(),
            p_()
        {}

        //! Constructor
        /*! 
         * \param M
         *        Matrix times vector application for which the equation 
         *          \f[ M x = b \f]
         *        is sovled for x.
         * \param Mprec
         *        Preconditioner matrix
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
            requires( NSL::Concept::isDerived<FermionMatrix<Type,LatticeType>,NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> )
        CGpreconditioned(
                FermionMatrix<Type,LatticeType> & M,
                FermionMatrix<Type,LatticeType> & Mprec,
                NSL::FermionMatrix::MatrixCombination matrixCombination,
                const NSL::RealTypeOf<Type> eps = 1e-12, const NSL::size_t maxIter = 10000) : 
            NSL::LinAlg::Solver<Type>(M,matrixCombination),
            innerCG_(Mprec, matrixCombination, eps, maxIter),
            errSq_(eps*eps),
            maxIter_(maxIter),
            x_(),
            t_(),
            r_(),
            z_(),
            p_()
        {}

        //! Constructor
        /*! 
         * \param M
         *        Matrix times vector application for which the equation 
         *          \f[ M x = b \f]
         *        is sovled for x.
         * \param Mprec
         *        Preconditioner matrix
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
            requires( NSL::Concept::isDerived<FermionMatrix<Type,LatticeType>,NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> )
        CGpreconditioned(
                FermionMatrix<Type,LatticeType> & M,
                std::function<NSL::Tensor<Type>(const NSL::Tensor<Type> &)> Mprec,
                NSL::FermionMatrix::MatrixCombination matrixCombination,
                const NSL::RealTypeOf<Type> eps = 1e-12, const NSL::size_t maxIter = 10000) : 
            NSL::LinAlg::Solver<Type>(M,matrixCombination),
            innerCG_(Mprec, eps, maxIter),
            errSq_(eps*eps),
            maxIter_(maxIter),
            x_(),
            t_(),
            r_(),
            z_(),
            p_()
        {}

        //! Constructor
        /*! 
         * \param M
         *        Matrix times vector application for which the equation 
         *          \f[ M x = b \f]
         *        is sovled for x.
         * \param Mprec
         *        Preconditioner matrix
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
            requires( NSL::Concept::isDerived<FermionMatrix<Type,LatticeType>,NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> )
        CGpreconditioned(
                FermionMatrix<Type,LatticeType> & M,
                std::function<NSL::Tensor<Type>(const NSL::Tensor<Type> &)> Mprec,
                const NSL::RealTypeOf<Type> eps = 1e-12, const NSL::size_t maxIter = 10000) : 
            NSL::LinAlg::Solver<Type>(M),
            innerCG_(Mprec, eps, maxIter),
            errSq_(eps*eps),
            maxIter_(maxIter),
            x_(),
            t_(),
            r_(),
            z_(),
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

        //! Apply CG
        /*!
         *  \param b, NSL::Tensor, RHS of the equation to be solved
         *
         *  \param x0, NSL::Tensor, Initial guess for x
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
        NSL::Tensor<Type> operator()(const NSL::Tensor<Type> & b, const NSL::Tensor<Type> & x0);

    private:

        // precision at which the algorithm is stopped
        const NSL::RealTypeOf<Type> errSq_;
        // maximum of iterations as fall back in case we don't converge
        const NSL::size_t maxIter_;

        // vector to store intermediate solution
        NSL::Tensor<Type> x_;
        // vector to store the result of M @ v 
        NSL::Tensor<Type> t_;
        // residual vector
        NSL::Tensor<Type> r_;
        // prec residual vector
        NSL::Tensor<Type> z_;
        // gradient vector
        NSL::Tensor<Type> p_;

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
        NSL::Tensor<Type> solve_(const NSL::Tensor<Type> & b);
        
        NSL::LinAlg::CG<Type> innerCG_;
}; // class CG
        
} //namespace NSL::LinAlg

#endif // NSL_CG_PREC_HPP
