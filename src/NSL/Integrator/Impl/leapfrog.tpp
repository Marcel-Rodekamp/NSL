#ifndef NSL_LEAPFROG_TPP
#define NSL_LEAPFROG_TPP

#include "../integrator.tpp"
#include "Configuration/Configuration.tpp"
#include "typePromotion.hpp"

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
	/*
	std::cout << "Pi half" << std::endl;
	for (int t = 0; t< 64; t++) {
	    std::cout << "{" << p["phi"][t*4+0].real() << ", " << p["phi"][t*4+0].imag() << " I, "
	                     << p["phi"][t*4+1].real() << ", " << p["phi"][t*4+1].imag() << " I, "
			     << p["phi"][t*4+2].real() << ", " << p["phi"][t*4+2].imag() << " I, "
			     << p["phi"][t*4+3].real() << ", " << p["phi"][t*4+3].imag() << " I}, "
	    << std::endl;
	 }
	 std::cout << std::endl;
	 */
			     
        // a bunch of full steps
        q += static_cast<TensorType>(stepSize_) * p;
	/*
	std::cout << "Q full" << std::endl;
	for (int t = 0; t< 64; t++) {
	    std::cout << "{" << q["phi"][t*4+0].real() << ", " << q["phi"][t*4+0].imag() << " I, "
	                     << q["phi"][t*4+1].real() << ", " << q["phi"][t*4+1].imag() << " I, "
			     << q["phi"][t*4+2].real() << ", " << q["phi"][t*4+2].imag() << " I, "
			     << q["phi"][t*4+3].real() << ", " << q["phi"][t*4+3].imag() << " I}, "
	    << std::endl;
	 }
	 std::cout << std::endl;
	 */

        for(NSL::size_t n = 0; n < numSteps_-1; ++n){
            p += static_cast<TensorType>(stepSize_) * this->action_.force(q);
	    /*
	    	std::cout << "Pi full" << std::endl;
	for (int t = 0; t< 64; t++) {
	    std::cout << "{" << p["phi"][t*4+0].real() << ", " << p["phi"][t*4+0].imag() << " I, "
	                     << p["phi"][t*4+1].real() << ", " << p["phi"][t*4+1].imag() << " I, "
			     << p["phi"][t*4+2].real() << ", " << p["phi"][t*4+2].imag() << " I, "
			     << p["phi"][t*4+3].real() << ", " << p["phi"][t*4+3].imag() << " I}, "
	    << std::endl;
	 }
	 std::cout << std::endl;
	 */
            q += static_cast<TensorType>(stepSize_) * p;
	    /*
	    	std::cout << "Q full" << std::endl;
	for (int t = 0; t< 64; t++) {
	    std::cout << "{" << q["phi"][t*4+0].real() << ", " << q["phi"][t*4+0].imag() << " I, "
	                     << q["phi"][t*4+1].real() << ", " << q["phi"][t*4+1].imag() << " I, "
			     << q["phi"][t*4+2].real() << ", " << q["phi"][t*4+2].imag() << " I, "
			     << q["phi"][t*4+3].real() << ", " << q["phi"][t*4+3].imag() << " I}, "
	    << std::endl;
	 }
	 std::cout << std::endl;
	 */
        }

        // final half step
        p += static_cast<TensorType>(0.5*stepSize_) * this->action_.force(q);

/*
	std::cout << "Pi half" << std::endl;
	for (int t = 0; t< 64; t++) {
	    std::cout << "{" << p["phi"][t*4+0].real() << ", " << p["phi"][t*4+0].imag() << " I, "
	                     << p["phi"][t*4+1].real() << ", " << p["phi"][t*4+1].imag() << " I, "
			     << p["phi"][t*4+2].real() << ", " << p["phi"][t*4+2].imag() << " I, "
			     << p["phi"][t*4+3].real() << ", " << p["phi"][t*4+3].imag() << " I}, "
	    << std::endl;
	 }
	 std::cout << std::endl;
*/
        return {q,p};
    }

    protected:
        double trajLength_;
        NSL::size_t numSteps_;
        double stepSize_;
};

}
#endif //NSL_LEAPFROG_TPP
