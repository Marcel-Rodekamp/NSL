#ifndef NSL_FERM_BASE_HPP
#define NSL_FERM_BASE_HPP

#include<vector>
#include "../Lattice/lattice.hpp"
#include "../assert.hpp"
#include "../complex.hpp"

#include "../Tensor/tensor.hpp"

namespace NSL::FermionMatrix{

//definition of class FermionMatrixBase

template<typename Type>
class FermionMatrixBase {

    public:

    //Declaration of methods methods M, M_dagger, MM_dagger and M

    virtual NSL::TimeTensor<Type> M(const NSL::TimeTensor<Type> & psi) = 0;
    virtual NSL::TimeTensor<Type> Mdagger(const NSL::TimeTensor<Type> & psi) = 0;
    virtual NSL::TimeTensor<Type> MMdagger(const NSL::TimeTensor<Type> & psi) = 0;
    virtual NSL::TimeTensor<Type> MdaggerM(const NSL::TimeTensor<Type> & psi) = 0;
//    virtual NSL::TimeTensor<Type> Mdagger(const NSL::TimeTensor<Type> & psi);
//    virtual NSL::TimeTensor<Type> MMdagger(const NSL::TimeTensor<Type> & psi);
//    virtual NSL::TimeTensor<Type> MdaggerM(const NSL::TimeTensor<Type> & psi);

//    NSL::complex<NSL::RT_extractor<Type>::value_type> detM();
//    NSL::complex<NSL::RT_extractor<Type>::value_type> detMdagger() {
//        return NSL::conj(this->detM());
//    }
//    NSL::complex<NSL::RT_extractor<Type>::value_type> detMdaggerM();
//    NSL::complex<NSL::RT_extractor<Type>::value_type> detMMdagger(){
//        return this->detMdaggerM();
//    }
//
//    NSL::complex<NSL::RT_extractor<Type>::value_type> logdetM();
//    NSL::complex<NSL::RT_extractor<Type>::value_type> logdetMdagger() {
//        //! \todo: How to do log as in detMdagger
//        //!        return NSL::conj(this->detM());
//        return static_cast<NSL::complex<RT_extractor<Type>::value_type>>(0);
//    }
//    NSL::complex<NSL::RT_extractor<Type>::value_type> logdetMdaggerM();
//    NSL::complex<NSL::RT_extractor<Type>::value_type> logdetMMdagger(){
//        //! \todo: How to do log as in detMdaggerM
//        //!        return this->detMdaggerM();
//        return static_cast<NSL::complex<RT_extractor<Type>::value_type>>(0);
//    }

    // constructors
    FermionMatrixBase() = delete;
    FermionMatrixBase(FermionMatrixBase<Type> &) = delete;
    FermionMatrixBase(FermionMatrixBase<Type> &&) = delete;

    FermionMatrixBase(NSL::Lattice::SpatialLattice<typename NSL::RT_extractor<Type>::value_type> * lat):
        Lat(lat)
    {}


    protected:

    NSL::Lattice::SpatialLattice<typename NSL::RT_extractor<Type>::value_type>* Lat;
};

}

#endif
