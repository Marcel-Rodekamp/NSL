#ifndef NSL_TWO_POINT_CORRELATION_FUNCTION_TPP
#define NSL_TWO_POINT_CORRELATION_FUNCTION_TPP

#include "concepts.hpp"
#include "parameter.tpp"
#include "../measure/hpp"

namespace NSL::Measure::Hubbard {

template<
    NSL::Concept::isNumber Type
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
class TwoPointCorrelator: public Measurement {
    public:
        TwoPointCorrelator(NSL::Parameter params, NSL::H5IO & h5):
            Measurement(h5),
            hfm_(params)
        {}

    void measure() override;

    protected:
        FermionMatrixType hfm_
};

template<
    NSL::Concept::isNumber Type
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoPointCorrelator<Type,LatticeType,FermionMatrixType>::measure(){
    NSL::Logger::info("Start Measuring Hubbard::TwoPointCorrelator");
}

}

#endif // NSL_TWO_POINT_CORRELATION_FUNCTION_TPP
