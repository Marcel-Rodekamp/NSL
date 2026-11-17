#ifndef NSL_INTEGRATOR_TPP
#define NSL_INTEGRATOR_TPP

#include "../Configuration.hpp"
#include "../Action.hpp"
#include "../concepts.hpp"
#include "../Tensor/Factory/like.tpp"

#include <cmath>
#include <concepts>

namespace NSL::Integrator {


// isTemplateDerived requires same template structure, this can not be available
//
//template<NSL::Concept::isTemplateDerived<NSL::Action::BaseAction> ... ActionTermTypes>
template<typename ... ActionTermTypes>
class Integrator{
    public:
    //! Base constructor of an Integrator
    /*! 
     * */
    Integrator(const NSL::Action::Action<ActionTermTypes...> & action):
        action_(action)
    {}

    //! Integrate a differential equation defined by force given in 'action'
    /*!
     * Numerical integrators apply the force to subsequently evolve 
     * a given configuration. This operator takes in the starting point
     * configuration of the evolution and returns the desired endpoint.
     * */
    template<NSL::Concept::isNumber ... TensorTypes>
    std::tuple<NSL::Configuration<TensorTypes...>, NSL::Configuration<TensorTypes ...> > 
        operator()(const NSL::Configuration<TensorTypes ...> & q, const NSL::Configuration<TensorTypes ...> & p);

    template<NSL::Concept::isNumber ... TensorTypes>
    std::tuple<NSL::Configuration<TensorTypes...>, NSL::Configuration<TensorTypes ...> > 
        operator()(const NSL::Configuration<TensorTypes ...> & q);

    protected:
    NSL::Action::Action<ActionTermTypes...> action_;
};

//! An integrator with a non-trivial mass matrix provides the momentum distribution
//! and the matching kinetic energy (e.g. ExactFourierAcceleration).
template<typename IntegratorType, typename Type>
concept hasMassMatrix = requires(IntegratorType & integrator, const NSL::Configuration<Type> & config) {
    { integrator.sampleMomentum(config) } -> std::same_as<NSL::Configuration<Type>>;
    { integrator.kineticEnergy(config) } -> std::convertible_to<Type>;
};

//! Momenta for an HMC trajectory starting at q: the integrator's distribution if it
//! has a mass matrix, otherwise real standard normal momenta (unit mass).
template<typename IntegratorType, NSL::Concept::isNumber Type>
NSL::Configuration<Type> sampleMomentum(IntegratorType & integrator, const NSL::Configuration<Type> & q){
    if constexpr (hasMassMatrix<IntegratorType,Type>) {
        return integrator.sampleMomentum(q);
    } else {
        NSL::Configuration<Type> momentum;
        for(auto & [key,field]: q){
            NSL::Tensor<Type> p = NSL::zeros_like(field);
            if constexpr (NSL::is_complex<Type>()) {
                // Tensor::randn() draws real and imaginary parts with unit variance each
                p.randn();
                p.imag()=0;
            } else {
                // Tensor::randn() scales the standard deviation by sqrt(2) (as appropriate for
                // complex tensors); real momenta need unit variance to match p^2/2
                p.randn(0., 1./std::sqrt(2.));
            }
            momentum[key] = p;
        }
        return momentum;
    }
}

//! Kinetic energy of the momenta p: the integrator's if it has a mass matrix,
//! otherwise p^2/2 (unit mass).
template<typename IntegratorType, NSL::Concept::isNumber Type>
Type kineticEnergy(IntegratorType & integrator, const NSL::Configuration<Type> & p){
    if constexpr (hasMassMatrix<IntegratorType,Type>) {
        return integrator.kineticEnergy(p);
    } else {
        Type K = static_cast<Type>(0);
        for(const auto & [key,field]: p){
            K += 0.5*(field * field).sum();
        }
        return K;
    }
}

}
#endif
