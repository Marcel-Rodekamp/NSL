#ifndef NSL_HUBBARD_FERMION_ACTION_TPP
#define NSL_HUBBARD_FERMION_ACTION_TPP

#include "../action.tpp"
#include "Configuration/Configuration.tpp"
#include "Tensor/Factory/like.tpp"
#include "concepts.hpp"
#include "FermionMatrix/fermionMatrix.hpp"

namespace NSL::Action {

//! Hubbard Gauge Action
/*!
 * Given a phi \f(\Phi\f) this action evaluates 
 * \f[ S(\Phi) = \frac{\Phi^2}{\delta U} \f]
 * where 
 *  - \f(\delta = \frac{\beta}{N_t}\f) is the lattice spacing
 *  - \f(\beta\f) is the inverse temperature
 *  - \f(N_t\f) is the number of time slices (troterization)
 *  - \f(U\f) is the on-site interaction of the Hubbard model Hamiltonian
 * */
template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType, 
    NSL::Concept::isNumber TensorType = Type
>
class HubbardFermionAction : 
    public BaseAction<Type, TensorType> 
{   
    public: 

    //! Parameter set for the Hubbard Gauge Action
    /*!
     * This parameters contain 
     * - `U` The on-site interaction
     * - `beta` The inverse temperature
     * - `Nt` The number of time slice (troterization)
     * */
    class Parameters { 
        public:
        // inverse temperature
	    const Type beta;

        // time slices
        const NSL::size_t Nt;

        // lattice
        LatticeType & lattice;

        Parameters(const Type & beta, const NSL::size_t & Nt, LatticeType & lattice):
            beta(beta), Nt(Nt), lattice(lattice), delta(beta/Nt) 
        {}

        // lattice spacing
        const Type delta;
    };

	HubbardFermionAction(const Parameters & params) : 
        BaseAction<Type, TensorType>(
            "phi"
        ),
        params_(params)
    {}

	HubbardFermionAction(const Parameters & params, const std::string & fieldName) : 
        BaseAction<Type, TensorType>(
            fieldName
        ),
        params_(params)
    {}

    // We import the eval/grad/force functions from the BaseAction such 
    // that we do not need to reimplement the Configuration based versions
    // We don't understand why this is not automatically done, probably due 
    // to BaseAction being an abstract base class
    using BaseAction<Type,TensorType>::eval;
    using BaseAction<Type,TensorType>::grad;
    using BaseAction<Type,TensorType>::force;

	Configuration<TensorType> force(const Tensor<TensorType>& phi);
	Configuration<TensorType> grad(const Tensor<TensorType>& phi);
	Type eval(const Tensor<TensorType>& phi);

    protected:
    Parameters params_;

    FermionMatrixType constexpr inline HFM(const NSL::Tensor<Type> & phi) {
        return FermionMatrixType(params_.lattice, phi, params_.beta);
    }

};

template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType, 
    NSL::Concept::isNumber TensorType
>
Type HubbardFermionAction<Type,LatticeType,FermionMatrixType,TensorType>::eval(const Tensor<TensorType>& phi){
    Type logDetMpMh = 0;

    // particle contribution
    auto Mp = this->HFM(phi);
    logDetMpMh+= Mp.logDetM();

    // hole contribution
    auto Mh = this->HFM(-phi);
    logDetMpMh+= Mh.logDetM();

    // The Fermi action has an additional - sign
    return -logDetMpMh;
}
	
template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType, 
    NSL::Concept::isNumber TensorType
>
Configuration<TensorType> HubbardFermionAction<Type,LatticeType,FermionMatrixType,TensorType>::force(const Tensor<TensorType>& phi){
    // The force has an explicit minus sign from the gradient
    return (-1)*this->grad(phi);
}

template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType, 
    NSL::Concept::isNumber TensorType
>
Configuration<TensorType> HubbardFermionAction<Type,LatticeType,FermionMatrixType,TensorType>::grad(const Tensor<TensorType>& phi){
    NSL::Configuration<TensorType> dS{{ this->configKey_, NSL::zeros_like(phi) }};

    // particle contribution
    auto Mp = HFM(phi);
    dS[this->configKey_]+= Mp.gradLogDetM();

    // hole contribution
    auto Mh = HFM(-phi);
    dS[this->configKey_]-= Mh.gradLogDetM();

    return dS;
}

} // namespace NSL::Action

#endif // NSL_HUBBARD_FERMION_ACTION_TPP
