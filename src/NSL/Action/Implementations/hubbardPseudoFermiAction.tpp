#ifndef NSL_HUBBARD_PSEUDO_FERMION_ACTION_TPP
#define NSL_HUBBARD_PSEUDO_FERMION_ACTION_TPP

#include "../action.tpp"
#include "Configuration/Configuration.tpp"
#include "Tensor/Factory/like.tpp"
#include "concepts.hpp"
#include "FermionMatrix/fermionMatrix.hpp"
#include "LinAlg.hpp"

namespace NSL::Action {

//! Hubbard Pseudo-Fermion Action //TODO Documentation
/*!
 * Given a phi \f(\Phi\f) and a xi \f(\Xi\f) this action evaluates 
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
class HubbardPseudoFermionAction: 
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

	HubbardPseudoFermionAction(const Parameters & params) : 
        BaseAction<Type, TensorType>(
            "phi"
        ),
        params_(params),
		xi(Tensor<TensorType>(params.Nt, params.lattice.sites())),
		xf(Tensor<TensorType>(params.Nt, params.lattice.sites())),
		eta(Tensor<TensorType>(params.Nt, params.lattice.sites()))
    {}

	HubbardPseudoFermionAction(const Parameters & params, const std::string & fieldName) : 
        BaseAction<Type, TensorType>(
            fieldName
        ),
        params_(params),
		xi(Tensor<TensorType>(params.Nt, params.lattice.sites())),
		xf(Tensor<TensorType>(params.Nt, params.lattice.sites())),
		eta(Tensor<TensorType>(params.Nt, params.lattice.sites()))
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

	void generate_eta(const Tensor<TensorType>& phi);

	Tensor<TensorType> xi,eta,xf;
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
void HubbardPseudoFermionAction<Type,LatticeType,FermionMatrixType,TensorType>::generate_eta(const Tensor<TensorType>& phi){
	xi.randn(0.7071067811865475244008444); // 1./std::sqrt(2);

        // particle matrix			       
        auto Mp = HFM(phi);

        // pseudofermion
    	eta = Mp.M(xi);

	xf = xi;
}

template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType, 
    NSL::Concept::isNumber TensorType
>
Type HubbardPseudoFermionAction<Type,LatticeType,FermionMatrixType,TensorType>::eval(const Tensor<TensorType>& phi){
    Type XX = xf.abs().sum();

    // The Fermi action has an additional - sign
    
    return XX;
}
	
template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType, 
    NSL::Concept::isNumber TensorType
>
Configuration<TensorType> HubbardPseudoFermionAction<Type,LatticeType,FermionMatrixType,TensorType>::force(const Tensor<TensorType>& phi){
    // The force has an explicit minus sign from the gradient
    return (-1)*this->grad(phi);
}


template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType, 
    NSL::Concept::isNumber TensorType
>
Configuration<TensorType> HubbardPseudoFermionAction<Type,LatticeType,FermionMatrixType,TensorType>::grad(const Tensor<TensorType>& phi){

    NSL::Configuration<TensorType> dS{{ this->configKey_, NSL::zeros_like(phi) }};

    // particle matrix
    auto Mp = HFM(phi);
    // hole matrix
    // auto Mh = HFM(-phi);

   
    // solve M M^dagger x = eta for x
    NSL::LinAlg::CG<NSL::complex<double>> invMMd(Mp, NSL::FermionMatrix::MMdagger);
    auto x = invMMd(eta); // = Mdag^-1 M^-1 eta
    // y = Mdag x = M^-1 eta
    xf = Mp.Mdagger(x); // I set this to xf

    // calculate y^dagger dM^dagger/dPhi x -> dMdPhi(NSL::LinAlg::conj(xf), Mh, x)
    dS[this->configKey_] -= 2*Mp.dMdPhi(NSL::LinAlg::conj(x), xf).real();

    // calculate x^dagger dM/dPhi y -> dMdPhi(x, Mp, y)
    // dS[this->configKey_] += Mp.dMdPhi(x, xf).conj();

    return dS;
}

} // namespace NSL::Action

#endif // NSL_HUBBARD_PSUEDO_FERMION_ACTION_TPP