#ifndef NSL_INTEGRATOR_TPP
#define NSL_INTEGRATOR_TPP

#include "../Configuration.hpp"
#include "../Action.hpp"
#include "../concepts.hpp"

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

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
class IntegratorBase{
    public:
    //! Base constructor of an Integrator
    /*! 
     * */
    IntegratorBase(const NSL::Action::BaseAction<Type, TensorType> & action):
        action_(action)
    {}

    //! Integrate a differential equation defined by force given in 'action'
    /*!
     * Numerical integrators apply the force to subsequently evolve 
     * a given configuration. This operator takes in the starting point
     * configuration of the evolution and returns the desired endpoint.
     * */
    std::tuple<NSL::Configuration<TensorType>, NSL::Configuration<TensorType> > 
        virtual operator()(const NSL::Configuration<TensorType> & q, const NSL::Configuration<TensorType> & p) = 0;

    std::tuple<NSL::Configuration<TensorType>, NSL::Configuration<TensorType> > 
        virtual operator()(const NSL::Configuration<TensorType> & q) = 0;

    protected:
    const NSL::Action::BaseAction<Type, TensorType> & action_;
};

}
#endif
