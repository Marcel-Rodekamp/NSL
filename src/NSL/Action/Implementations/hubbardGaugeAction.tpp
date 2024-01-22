#ifndef NSL_HUBBARD_GAUGE_ACTION_TPP
#define NSL_HUBBARD_GAUGE_ACTION_TPP

#include "../action.tpp"
#include "hubbard.tpp"

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
template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType = Type>
class HubbardGaugeAction : 
    public BaseAction<Type, TensorType> 
{   
    public: 

	HubbardGaugeAction(NSL::Parameter & params) : 
        BaseAction<Type, TensorType>(
            "phi"
        ),
        params_(params),
        Utilde_(NSL::Hubbard::tilde<Type>(params,"U"))
    {}

	HubbardGaugeAction(NSL::Parameter & params,const std::string & fieldName) : 
        BaseAction<Type, TensorType>(
            fieldName
        ),
        params_(params),
        Utilde_(NSL::Hubbard::tilde<Type>(params,"U")) 
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
    NSL::Parameter params_;
    Type Utilde_;
};

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
Type HubbardGaugeAction<Type, TensorType>::eval(const Tensor<TensorType>& phi){
    return (phi * phi).sum() / ( 2 * Utilde_ ) ;
}
	
template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
Configuration<TensorType> HubbardGaugeAction<Type, TensorType>::force(const Tensor<TensorType>& phi){
    return Configuration<Type>{{this->configKey_, phi /(- Utilde_)}};
}

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
Configuration<TensorType> HubbardGaugeAction<Type, TensorType>::grad(const Tensor<TensorType>& phi){
    return Configuration<Type>{{this->configKey_, phi / Utilde_}};
}

} // namespace NSL::Action

#endif // NSL_HUBBARD_GAUGE_ACTION_TPP
