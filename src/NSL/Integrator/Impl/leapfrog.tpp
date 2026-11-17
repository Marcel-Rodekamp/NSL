#ifndef NSL_LEAPFROG_TPP
#define NSL_LEAPFROG_TPP

#include "../integrator.tpp"
#include "Configuration/Configuration.tpp"
#include "typePromotion.hpp"

#include <stdexcept>

namespace NSL::Integrator {


//template<NSL::Concept::isTemplateDerived<NSL::Action::BaseAction> ... ActionTermTypes>
template<typename ... ActionTermTypes>
class Leapfrog: Integrator<ActionTermTypes...> {
    public:
    //! Constructor of the leapfrog 
    /*! 
     * */
    Leapfrog(const NSL::Action::Action<ActionTermTypes...> & action, 
             const double & trajectoryLength,
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
    template<NSL::Concept::isNumber TensorType>
    std::tuple<NSL::Configuration<TensorType>, NSL::Configuration<TensorType> > operator()(
        const NSL::Configuration<TensorType> & q_, const NSL::Configuration<TensorType> & p_
    ){
        
        // deep copy to generate a new configuration
        NSL::Configuration<TensorType> q (q_,true);
        NSL::Configuration<TensorType> p (p_,true);

        // first half step 
        p += static_cast<TensorType>(0.5*stepSize_)* this->action_.force(q);
			     
        // a bunch of full steps
        q += static_cast<TensorType>(stepSize_) * p;

        for(NSL::size_t n = 0; n < numSteps_-1; ++n){
            p += static_cast<TensorType>(stepSize_) * this->action_.force(q);
            q += static_cast<TensorType>(stepSize_) * p;
        }

        // final half step
        p += static_cast<TensorType>(0.5*stepSize_) * this->action_.force(q);

        return {q,p};
    }

    //! Number of MD steps per trajectory; the trajectory length stays fixed (used by NSL::MCMC::Autotune)
    void setNumberSteps(NSL::size_t numberSteps){
        if (numberSteps < 1) {
            throw std::invalid_argument("NSL::Integrator: the number of MD steps must be at least 1");
        }
        const double sign = (stepSize_ < 0) ? -1. : 1.;
        numSteps_ = numberSteps;
        stepSize_ = sign * trajLength_ / static_cast<double>(numberSteps);
    }
    NSL::size_t numberSteps() const { return numSteps_; }
    double trajectoryLength() const { return trajLength_; }

    protected:
        double trajLength_;
        NSL::size_t numSteps_;
        double stepSize_;
};

}
#endif //NSL_LEAPFROG_TPP
