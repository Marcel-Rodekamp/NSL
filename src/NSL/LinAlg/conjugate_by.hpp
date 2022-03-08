#ifndef NANOSYSTEMLIBRARY_CONJUGATE_BY_HPP
#define NANOSYSTEMLIBRARY_CONJUGATE_BY_HPP

#include <torch/torch.h>
#include "../LinAlg.hpp"

//! \file conjugate_by.hpp
//! Conjugation by a matrix
/*!
 *  Using the matrix S in a similarity transformation of a matrix M is given by
 *     inverse(S) M S
 *  where the matrix multiplication is implied.
 *
 *  Naively, computing the similarity transformation requires inverting S, which is often prohibitive.
 *  However, in some special cases below the similarity transformation may be computed quickly.
 *
 * */

namespace NSL{
    namespace LinAlg {

        //! Conjugation by any invertible matrix.
        /*!
         * \param S conjugation matrix
         * \param M matrix being conjugated
         * \returns inverse(S) M S; result shares a type with M.
         *
         * Assumes that S and M are compatible square matrices.
         *
         * \todo May need a more flexible implementation where the return type is deduced or auto?
         * */
        template <typename Type, typename OtherType>
        NSL::Tensor<OtherType> conjugate_by(const NSL::Tensor<Type> & S, const NSL::Tensor<OtherType> & M){
            // Implementation note: rather than actually invert S, it is much more numerically stable to solve SX = MS.
            return NSL::Tensor<OtherType>( torch::linalg::solve(to_torch(S), to_torch(NSL::LinAlg::mat_mul(M,S)) ));
        }

        //! Conjugation by a unitary matrix; leverages the fact that \f$U^\dagger = U^{-1}\f$.
        /*!
         * \param U a unitary matrix
         * \param M a matrix to be conjugated
         * \returns U.adjoint() @ M @ U; result shares a type with M.
         *
         * Assumes that U and M are compatible square matrices.
         *
         * \todo May need a more flexible implementation where the return type is deduced or auto?
         * */
        template <typename Type, typename OtherType>
        NSL::Tensor<OtherType> conjugate_by_unitary(const NSL::Tensor<Type> & U, const NSL::Tensor<OtherType> & M){
            auto u = U;
            auto result(M);
            result = NSL::LinAlg::mat_mul(result, u);
            result = mat_mul(u.adjoint(), result);
            return result;
        }

        //! Conjugation by an orthogonal matrix; leverages the fact that \f$O^\top = O^{-1}\f$.
        /*!
         * \param O an orthogonal matrix
         * \param M a matrix to be conjugated
         * \returns O.transpose() @ M @ O; result shares a type with M.
         *
         * Assumes that O and M are compatible square matrices.
         *
         * \todo May need a more flexible implementation where the return type is deduced or auto?
         * */
        template <typename RealType, typename OtherRealType>
        NSL::Tensor<OtherRealType> conjugate_by_orthogonal(
                const NSL::Tensor<RealType, RealType> & O,
                const NSL::Tensor<OtherRealType, OtherRealType> & M
                ){
            return conjugate_by_unitary(O, M);
        }
        
    } // namespace LinAlg
} // namespace NSL

#endif //NANOSYSTEMLIBRARY_CONJUGATE_BY_HPP
