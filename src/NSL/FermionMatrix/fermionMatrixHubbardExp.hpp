#ifndef NANOSYSTEMLIBRARY_FERMIONMATRIXHUBBARDEXP_HPP
#define NANOSYSTEMLIBRARY_FERMIONMATRIXHUBBARDEXP_HPP

#include "LinAlg/mat_vec.hpp"
#include "fermionMatrixBase.hpp"
#include "../LinAlg/mat_conj.hpp"
#include "../LinAlg/mat_mul.hpp"
#include "../LinAlg/mat_exp.hpp"
#include "../LinAlg/mat_trans.hpp"
#include "LinAlg/mat_inv.hpp"
#include "../LinAlg/det.hpp"
#include "../LinAlg/exp.hpp"
#include "../Tensor/Matrices/matricesBase.hpp"

namespace NSL::FermionMatrix {
template<typename Type>

//! \todo: Remove FermionMatrix from the name
//! \todo: Remove Hubbard from the name and make it a namespace
class FermionMatrixHubbardExp : public FermionMatrixBase<Type> {

    public:
    FermionMatrixHubbardExp() = delete;

    //!
    /*!
     *
     * \todo: SpatialLattice accepts only real type template arguments
     * */
    FermionMatrixHubbardExp(NSL::Lattice::SpatialLattice<typename NSL::RT_extractor<Type>::value_type> *lat, const NSL::TimeTensor<Type> &phi):
        FermionMatrixBase<Type>(lat),
        phi_(phi),
        phiExp_(NSL::LinAlg::exp(phi*NSL::complex<typename NSL::RT_extractor<Type>::value_type>(0,1)))
        

    {}

    NSL::TimeTensor<Type> M(const NSL::TimeTensor<Type> & psi) override;
    NSL::TimeTensor<Type> Mdagger(const NSL::TimeTensor<Type> & psi) override;
    NSL::TimeTensor<Type> MMdagger(const NSL::TimeTensor<Type> & psi) override;
    NSL::TimeTensor<Type> MdaggerM(const NSL::TimeTensor<Type> & psi) override;
    Type logDetM() override;
    Type logDetMdagger() override;
    /*NSL::complex<Type> logDetM(const NSL::TimeTensor<Type> & psi) override;
    NSL::TimeTensor<Type> NSL::FermionMatrix::FermionMatrixHubbardExp<Type>::F() override;
    */
    protected:
    NSL::Tensor<Type> phi_;
    NSL::Tensor<Type> phiExp_;

    private:
    NSL::TimeTensor<Type> F_(const NSL::TimeTensor<Type> & psi);

};
} // namespace FermionMatrix

#endif //NANOSYSTEMLIBRARY_FERMIONMATRIXHUBBARDEXP_HPP
