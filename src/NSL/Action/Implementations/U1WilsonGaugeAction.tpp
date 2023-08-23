#ifndef NSL_SCHWINGER_MODEL_ACTION_TPP
#define NSL_SCHWINGER_MODEL_ACTION_TPP

#include "../action.tpp"
#include "Configuration/Configuration.tpp"
#include "U1.hpp"

namespace NSL::Action::U1 {

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
template<NSL::Concept::isNumber Type>
class WilsonGaugeAction : public BaseAction<Type, Type> {   
    public: 

	WilsonGaugeAction(NSL::Parameter & params) : 
        BaseAction<Type, Type>("U"),
        params_(params),
        beta_(params["beta"])
    {}

	WilsonGaugeAction(NSL::Parameter & params, const std::string & fieldName) : 
        BaseAction<Type, Type>(fieldName),
        params_(params),
        beta_(params["beta"])
    {}

    // We import the eval/grad/force functions from the BaseAction such 
    // that we do not need to reimplement the Configuration based versions
    // We don't understand why this is not automatically done, probably due 
    // to BaseAction being an abstract base class
    using BaseAction<Type,Type>::eval;
    using BaseAction<Type,Type>::grad;
    using BaseAction<Type,Type>::force;

	Configuration<Type> force(const Tensor<Type>& phi);
	Configuration<Type> grad(const Tensor<Type>& phi);
	Type eval(const Tensor<Type>& phi);

    protected:
    NSL::Parameter params_;
    Type beta_;

}; // class WilsonGaugeAction

template<NSL::Concept::isNumber Type>
Type WilsonGaugeAction<Type>::eval(const Tensor<Type>& phi){
    Type S = 0;

    for( NSL::size_t mu = 0; mu < params_["dim"].to<NSL::size_t>(); ++mu){
        for( NSL::size_t nu = mu+1; nu < params_["dim"].to<NSL::size_t>(); ++nu){
            NSL::Tensor<Type> P = NSL::U1::plaquette(phi,mu,nu);
            //sum_{x} P_mu,nu + P_mu,nu^{-1}
            S += ( 1 - 0.5*(P + 1./P)  ).sum();
        }
    }

    return beta_*S;
}

template<NSL::Concept::isNumber Type>
Configuration<Type> WilsonGaugeAction<Type>::grad(const Tensor<Type>& phi){
    NSL::complex<NSL::RealTypeOf<Type>> I{0,1}; 

    // calculate staple K_\mu(x) from P_{\mu\nu}(x) = U_{\mu}(x) K_{\mu}(x)
    auto staple = phi*NSL::U1::sumAdjacentStaples(phi);
    // as everything commutes the inverse is just the inverse of this.

    return NSL::Configuration<NSL::complex<NSL::RealTypeOf<Type>>> {{this->configKey_, 
        beta_*staple.imag()
    }};
}

template<NSL::Concept::isNumber Type>
Configuration<Type> WilsonGaugeAction<Type>::force(const Tensor<Type>& phi){
    return (-1)*this->grad(phi);
}

} // namespace NSL::Action::U1

#endif // NSL_HUBBARD_FERMION_ACTION_TPP
