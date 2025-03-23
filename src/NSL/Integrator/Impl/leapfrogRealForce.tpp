#ifndef NSL_LEAPFROG_REAL_FORCE_TPP
#define NSL_LEAPFROG_REAL_FORCE_TPP

#include "../integrator.tpp"
#include "Configuration/Configuration.tpp"
#include "typePromotion.hpp"

#define USE_NVTX
#include "profiling.hpp"

namespace NSL::Integrator {


//template<NSL::Concept::isTemplateDerived<NSL::Action::BaseAction> ... ActionTermTypes>
template<typename ... ActionTermTypes>
class LeapfrogRealForce: Integrator<ActionTermTypes...> {
    public:
    //! Constructor of the leapfrog 
    /*! 
     * */
    LeapfrogRealForce(const NSL::Action::Action<ActionTermTypes...> & action, 
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

    //! Integrate a differential equation defined by the real part of the force given in 'action'.
    // 
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
    PUSH_RANGE("Force", 3);
	auto tmp = this->action_.force(q);
    POP_RANGE;
	for(auto & [key,field]: tmp){
		 field.imag() = 0; // ??
	}
	p += static_cast<TensorType>(0.5*stepSize_)*tmp;
        //p += static_cast<TensorType>(0.5*stepSize_)* NSL::real(this->action_.force(q));

        // a bunch of full steps
        q += static_cast<TensorType>(stepSize_) * p;

        for(NSL::size_t n = 0; n < numSteps_-1; ++n){
	    PUSH_RANGE("Force", 3);
        tmp = this->action_.force(q);
        POP_RANGE;
	    for(auto & [key,field]: tmp){
		 field.imag() = 0; // ??
	    }
	    p += static_cast<TensorType>(stepSize_)*tmp;
            //p += static_cast<TensorType>(stepSize_) * NSL::real(this->action_.force(q));
            q += static_cast<TensorType>(stepSize_) * p;
        }

        // final half step
    PUSH_RANGE("Force", 3);
	tmp = this->action_.force(q);
    POP_RANGE;
	for(auto & [key,field]: tmp){
		 field.imag() = 0; // ??
	}
	p += static_cast<TensorType>(0.5*stepSize_)*tmp;
        //p += static_cast<TensorType>(0.5*stepSize_) * NSL::real(this->action_.force(q));

        return {q,p};
    }

    protected:
        double trajLength_;
        NSL::size_t numSteps_;
        double stepSize_;
};

}

#undef USE_NVTX
#endif //NSL_LEAPFROG_REAL_FORCE_TPP
