#ifndef NANOSYSTEMLIBRARY_FERMIONMATRIXHUBBARDEXP_HPP
#define NANOSYSTEMLIBRARY_FERMIONMATRIXHUBBARDEXP_HPP

#include "../LinAlg/mat_vec.hpp"
#include "fermionMatrixBase.hpp"

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
        phi_(phi)
    {}

    NSL::TimeTensor<Type> M(const NSL::TimeTensor<Type> & psi) override;

    protected:
    NSL::Tensor<Type> phi_;

    private:
    NSL::TimeTensor<Type> F_(const NSL::TimeTensor<Type> & psi);

};
} // namespace FermionMatrix

#endif //NANOSYSTEMLIBRARY_FERMIONMATRIXHUBBARDEXP_HPP
