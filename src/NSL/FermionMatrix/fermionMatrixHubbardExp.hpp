#ifndef NANOSYSTEMLIBRARY_FERMIONMATRIXHUBBARDEXP_HPP
#define NANOSYSTEMLIBRARY_FERMIONMATRIXHUBBARDEXP_HPP

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
    FermionMatrixHubbardExp() = delete;

    //!
    /*!
     *
     * \todo: SpatialLattice accepts only real type template arguments
     * */
    FermionMatrixHubbardExp(NSL::Lattice::SpatialLattice<typename NSL::RT_extractor<Type>::value_type> *lat,  const NSL::TimeTensor<Type> &phi, double beta = 1.0 ):
        FermionMatrixBase<Type>(lat),
        phi_(phi),
        delta_(beta/phi.shape(0)),
        phiExp_(NSL::LinAlg::exp(phi*NSL::complex<typename NSL::RT_extractor<Type>::value_type>(0,1)))
        

    {}

    NSL::TimeTensor<Type> M(const NSL::TimeTensor<Type> & psi) override;
    NSL::TimeTensor<Type> Mdagger(const NSL::TimeTensor<Type> & psi) override;
    NSL::TimeTensor<Type> MMdagger(const NSL::TimeTensor<Type> & psi) override;
    NSL::TimeTensor<Type> MdaggerM(const NSL::TimeTensor<Type> & psi) override;
    Type logDetM() override;

    protected:
    NSL::Tensor<Type> phi_;
    NSL::Tensor<Type> phiExp_;
    NSL::RT_extractor<Type>::value_type delta_;

    private:
    NSL::TimeTensor<Type> F_(const NSL::TimeTensor<Type> & psi);

};
} // namespace FermionMatrix

#endif //NANOSYSTEMLIBRARY_FERMIONMATRIXHUBBARDEXP_HPP
