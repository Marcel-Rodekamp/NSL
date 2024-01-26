#ifndef NSL_RUNGE_KUTTA_2_TPP
#define NSL_RUNGE_KUTTA_2_TPP

#include "../integrator.tpp"
#include "Configuration/Configuration.tpp"
#include "typePromotion.hpp"

namespace NSL::Integrator {


//template<NSL::Concept::isTemplateDerived<NSL::Action::BaseAction> ... ActionTermTypes>
template<typename ... ActionTermTypes>
class RungeKutta2: Integrator<ActionTermTypes...> {
    public:
    //! Constructor of the RungeKutta 
    /*! 
     * */
    RungeKutta2(const NSL::Action::Action<ActionTermTypes...> & action, 
        const double & maxTime,
        const NSL::size_t & numberSteps,
        bool conjugategrad = false
    ):
            Integrator<ActionTermTypes...>(action),
            numSteps_(numberSteps),
            conjugateGrad_(conjugategrad),
            stepSize_(static_cast<double>(maxTime)/static_cast<double>(numberSteps)) 
    {}

    //! Integrate a differential equation defined by grad given in 'action' or by conj(grad) 
    //! if conjugategrad = True;
    /*!
     * */
    template<NSL::Concept::isNumber Type>
    NSL::Configuration<Type> operator()(const NSL::Configuration<Type> & q_){

        // deep copy to generate a new configuration
        NSL::Configuration<Type> q (q_,true);

        for(NSL::size_t n = 0; n < numSteps_; ++n){
            // Compute k1 = f(q_n)
            NSL::Configuration<Type> k1 = stepSize_ * grad_(q);

            // Compute k2 = f( q_n + (eps/2)*k1 )
            NSL::Configuration<Type> k2 = stepSize_ * grad_( q + 0.5 * k1 );
            
            // compute q_{n+1}
            q += k2; 
        }

        return q;
    }

    //! Update step size.
    /*! 
     * Use this function with care as it does NOT update the number of steps
     * your end point time thus changes
     * */
    double & stepSize(){return stepSize_;}

    //! Update Number of Steps.
    /*! 
     * Use this function with care as it does NOT update the step size
     * your end point time thus changes!
     * */
    NSL::size_t & numSteps(){return numSteps_;}

    protected:
        template<NSL::Concept::isNumber Type>
        inline NSL::Configuration<Type> grad_( NSL::Configuration<Type>  q ){
            NSL::Configuration<Type> F = this->action_.force(q);
            
            if (conjugateGrad_){
                for (auto & [key,field]: F){
                    field = field.conj();
                }
            }

            return F;
        }

        NSL::size_t numSteps_;
        double stepSize_;
        bool conjugateGrad_;
};

}
#endif //NSL_RUNGE_KUTTA_2_TPP
