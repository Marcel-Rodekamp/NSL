#ifndef NSL_LEAPFROG_TPP
#define NSL_LEAPFROG_TPP

#include "../integrator.tpp"
#include "Configuration/Configuration.tpp"
#include "typePromotion.hpp"

namespace NSL::Integrator {


template<NSL::Concept::isTemplateDerived<NSL::Action::BaseAction> ... ActionTermTypes>
class Leapfrog: Integrator<ActionTermTypes...> {
    public:
    //! Constructor of the leapfrog 
    /*! 
     * */
    Leapfrog(const NSL::Action::Action<ActionTermTypes...> & action, 
             const NSL::size_t & trajectoryLength,
             const NSL::size_t & numberSteps,
             bool backward = false):
            Integrator<ActionTermTypes...>(action),
            trajLength_(trajectoryLength),
            numSteps_(numberSteps),
            stepSize_(static_cast<double>(trajectoryLength)/static_cast<double>(numberSteps)) {
        if (backward){
            stepSize_ *= -1;
        }
    }

    //! Integrate a differential equation defined by force given in 'action'
    /*!
     * */
    template<NSL::Concept::isNumber ... TensorTypes>
    std::tuple<NSL::Configuration<TensorTypes...>, NSL::Configuration<TensorTypes ...> > operator()(
        const NSL::Configuration<TensorTypes ...> & q_, const NSL::Configuration<TensorTypes ...> & p_
    ){

        // deep copy to generate a new configuration
        NSL::Configuration<TensorTypes...> q (q_,true);
        NSL::Configuration<TensorTypes...> p (p_,true);
        
        //! \todo: I should add proper overloads to NSL::Configuration
        auto stepSize = static_cast<NSL::complex<double>>(stepSize_);

        // first half step 
        q += 0.5 * stepSize * p;

        // a bunch of full steps
        for(NSL::size_t n = 0; n < numSteps_; ++n){
            p+= stepSize * this->action_.force(q);
            q+= stepSize * p;
        }

        // final half step 
        p += 0.5 * stepSize * this->action_.force(q);
        q += 0.5 * stepSize * p;
        
        return {q,p};
    }

    protected:
        NSL::size_t trajLength_;
        NSL::size_t numSteps_;
        double stepSize_;
};

}
#endif //NSL_LEAPFROG_TPP
