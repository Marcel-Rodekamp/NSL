#ifndef NSL_HUBBARD_FERMION_ACTION_TPP
#define NSL_HUBBARD_FERMION_ACTION_TPP

#include "../action.tpp"
#include "Configuration/Configuration.tpp"
#include "Tensor/Factory/like.tpp"
#include "concepts.hpp"
#include "FermionMatrix/fermionMatrix.hpp"
#include "hubbard.tpp"

#include <memory>

namespace NSL::FermionMatrix {

//! A charge (particle/hole) basis fermion matrix is populated per NSL::Hubbard::Species
//! (e.g. HubbardExp: the auxiliary field enters as exp(±iφ)).
template<typename FermionMatrixType, typename Type>
concept ChargeBasis = requires(FermionMatrixType m, const NSL::Tensor<Type> & phi) {
    m.populate(phi, NSL::Hubbard::Species::Particle);
};

//! A spin basis fermion matrix is populated per NSL::Hubbard::Spin
//! (e.g. HubbardExpSpinBasis: the auxiliary field enters as exp(±φ)).
template<typename FermionMatrixType, typename Type>
concept SpinBasis = requires(FermionMatrixType m, const NSL::Tensor<Type> & phi) {
    m.populate(phi, NSL::Hubbard::Spin::Up);
};

} // namespace NSL::FermionMatrix

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

	HubbardFermionAction(LatticeType & lattice, NSL::Parameter & params, const std::string & fieldName) : 
        BaseAction<Type, TensorType>(fieldName),
        params_(params),
        hfm_(lattice,params)
    {}

	HubbardFermionAction(LatticeType & lattice, NSL::Parameter & params): 
	    HubbardFermionAction<Type,LatticeType,FermionMatrixType>(lattice,params,"phi")
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

    //! Evaluate the fermion determinants at phi + shift instead of phi (eval, grad and force).
    /*!
     * The shift is shared by all copies of this action (an NSL::Action::Action and the integrators keep
     * copies), so it can be changed between HMC trajectories, e.g. by NSL::MCMC::BridgeLink. The gradient
     * is taken with respect to phi at fixed shift. nullptr (the default) means no shift.
     * */
    void setShift(std::shared_ptr<Tensor<TensorType>> shift) { shift_ = std::move(shift); }
    const std::shared_ptr<Tensor<TensorType>> & shift() const { return shift_; }

    FermionMatrixType hfm_;

    protected:
    NSL::Parameter params_;
    std::shared_ptr<Tensor<TensorType>> shift_;

    //! the field the determinants are evaluated at
    Tensor<TensorType> shifted_(const Tensor<TensorType>& phi) const { return shift_ ? Tensor<TensorType>(phi + *shift_) : phi; }

}; // class HubbardFermiAction

template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType, 
    NSL::Concept::isNumber TensorType
>
Type HubbardFermionAction<Type,LatticeType,FermionMatrixType,TensorType>::eval(const Tensor<TensorType>& phi){
    Type logDetMpMh = 0;
    const Tensor<TensorType> phiS = shifted_(phi);

    // The basis is a property of the fermion-matrix type, so it is resolved at
    // compile time (see ChargeBasis/SpinBasis above).
    if constexpr (NSL::FermionMatrix::ChargeBasis<FermionMatrixType,Type>) {

     // particle contribution
     hfm_.populate(phiS, NSL::Hubbard::Species::Particle);
     logDetMpMh+= hfm_.logDetM();

     // hole contribution
     hfm_.populate(phiS, NSL::Hubbard::Species::Hole);
     logDetMpMh+= hfm_.logDetM();

    } else {
     static_assert(NSL::FermionMatrix::SpinBasis<FermionMatrixType,Type>,
        "HubbardFermionAction: fermion matrix must provide populate(phi, Species) (charge basis) or populate(phi, Spin) (spin basis)");

     // spin up contribution
     hfm_.populate(phiS, NSL::Hubbard::Spin::Up);
     logDetMpMh+= hfm_.logDetM();

     // spin down contribution
     hfm_.populate(phiS, NSL::Hubbard::Spin::Down);
     logDetMpMh+= hfm_.logDetM();

    }

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
    const Tensor<TensorType> phiS = shifted_(phi);

    if constexpr (NSL::FermionMatrix::ChargeBasis<FermionMatrixType,Type>) {

     if(hfm_.bipartite_ && hfm_.mu_.real() == 0 && hfm_.mu_.imag() == 0) {;
        // 2x real part of particle contribution

        hfm_.populate(phiS, NSL::Hubbard::Species::Particle);
        dS[this->configKey_]+= 2*hfm_.gradLogDetM().real();

     } else {

       // particle contribution
       hfm_.populate(phiS, NSL::Hubbard::Species::Particle);
       dS[this->configKey_]+= hfm_.gradLogDetM();

       // hole contribution
       hfm_.populate(phiS, NSL::Hubbard::Species::Hole);
       dS[this->configKey_]-= hfm_.gradLogDetM();

     }
    } else {
      static_assert(NSL::FermionMatrix::SpinBasis<FermionMatrixType,Type>,
        "HubbardFermionAction: fermion matrix must provide populate(phi, Species) (charge basis) or populate(phi, Spin) (spin basis)");

      // spin up contribution
      hfm_.populate(phiS, NSL::Hubbard::Spin::Up);
      dS[this->configKey_]+= hfm_.gradLogDetM();

      // spin down contribution
      hfm_.populate(phiS, NSL::Hubbard::Spin::Down);
      dS[this->configKey_]-= hfm_.gradLogDetM();

   }

     return dS;
}

} // namespace NSL::Action

#endif // NSL_HUBBARD_FERMION_ACTION_TPP
