#ifndef NSL_SOLVER_APPLICATION_TYPE_TPP
#define NSL_SOLVER_APPLICATION_TYPE_TPP

//! \file LinAlg/Solver/applicationType.tpp

namespace NSL::FermionMatrix {

//! Defines the matrix application type.
/*!
 * Some solvers like `NSL::LinAlg::CG` depend  on the hermiticiy - or more 
 * accurate, symmetric and positive definiteness in this case - of the used
 * fermion matrix. This is trivally given for the combination \f$ M^\dagger M, \, MM^\dagger \f$
 * To allow the flexibility of choising what ever combination defined in 
 * `NSL::FermionMatrix::FermionMatrix` this enumeration identifies the desired
 * case for the solver without passing a lambda to the solver construction.
 *
 * */
enum MatrixCombination { M, Mdagger, MMdagger, MdaggerM };

} // namespace NSL::FermionMatrix

#endif //NSL_SOLVER_APPLICATION_TYPE_TPP
