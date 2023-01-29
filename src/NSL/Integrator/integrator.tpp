#ifndef NSL_INTEGRATOR_TPP
#define NSL_INTEGRATOR_TPP

#include "../Configuration.hpp"
#include "../Action.hpp"
#include "../concepts.hpp"

namespace NSL::Integrator {


template<NSL::Concept::isTemplateDerived<NSL::Action::BaseAction> ... ActionTermTypes>
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

    protected:
    NSL::Action::Action<ActionTermTypes...> action_;
};

}
#endif
