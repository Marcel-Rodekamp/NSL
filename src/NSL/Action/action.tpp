#ifndef NANOSYSTEMLIBRARY_ACTION_HPP
#define NANOSYSTEMLIBRARY_ACTION_HPP

/*! \file action.hpp
 *  Classes for different actions.
 *	
 * Holds on the fermion matrix
 * Holds on the action parameters
 * Provides `operator()` computing the acion of a given config
 * Provides `eval` being the same as `operator()`
 * Provides `force` computing the force of a given config
 * Provides `grad` providing the derivative of the action in respect to a given config
 **/

#include "../Configuration.hpp"
#include<tuple>
#include<concepts>
#include <type_traits>

namespace NSL::Action {

template<class Action> 
struct params;


/*! A base class for actions.
 *      Offers the default functionality of actions and 
 *  	is acting as the parent class for the specific actions.
 **/

template<typename ... TensorTypes>
class BaseAction{
	public:
	virtual Configuration<TensorTypes...> force(const Tensor<TensorTypes>&... fields) = 0;
	virtual Configuration<TensorTypes...> grad(const Tensor<TensorTypes>&... fields) = 0;
	virtual complex<double> eval(const Tensor<TensorTypes>& ... fields) = 0;
};

/*! A wrapper class for Action implementations.
 * Handles the interaction between configurations and actions
 **/

template<class ActionImp> 
class Action {
	private:
	ActionImp Act;			//? derive from ActionImp instead?
	std::string key;		//TODO make general for actions with multiple fields
	
	public:
	Action(std::string pkey, params<ActionImp> params) :Act(ActionImp(params)),key(pkey) {}

	template<class Configuration>
	Configuration force(Configuration & config, Configuration & force, bool add = false){
		if(!add) force.zero();
		for(auto fName: config.fieldNames()){
			if (std::string(fName) == key){			// later there will be multiple keys
				force.field(key) += (Act.force(config.field(key))).field("force");	//
			}
		}
		return force;
	};

	template<class Configuration>
	Configuration grad(Configuration & config, Configuration & grad, bool add = false){
		if(!add) grad.zero();
		for(auto fName: config.fieldNames()){
			if (std::string(fName) == key){			// later there will be multiple keys
				grad.field(key) += (Act.grad(config.field(key))).field("grad");	//
			}
		}
		return grad;
	};

	template<class Configuration>
	complex<double> eval(Configuration& config) { 
		return Act.eval(config.field(key)); 
	};						//TODO implement for non homogeneous configs
};

/*! Container class for modular actions.
 * Offers the same functionality as the Action class.
 * It contains multiple actions and sums up their outputs.
 **/

template<class ...Actions>
class SumAction {
	private:
	std::tuple<Actions...>  Summands;
	
	template<int I, class Configuration>
    void recursive_sum_eval(complex<double> & sum, Configuration & config){
        if constexpr (I < sizeof...(Actions)){
            sum += std::get<I>(Summands).eval(config);
            recursive_sum_eval<I+1>(sum, config);
        } 
    }

	template<int I, class Configuration>
    void recursive_sum_force(Configuration & config, Configuration & force){
        if constexpr (I < sizeof...(Actions)){
			force = std::get<I>(Summands).force(config, force, true);
			recursive_sum_force<I+1>(config, force);
        } 
    }

	template<int I, class Configuration>
    void recursive_sum_grad(Configuration & config, Configuration & grad){
        if constexpr (I < sizeof...(Actions)){
			grad = std::get<I>(Summands).grad(config, grad, true);
			recursive_sum_grad<I+1>(config, grad);
        } 
    }
	
	public:
	SumAction(Actions ... pSummands) :Summands(pSummands ...){}

	template<class Configuration>
    complex<double> eval(Configuration & config){
		complex<double> sum = 0;
        recursive_sum_eval<0>(sum, config);

        return sum;
    }

	template<class Configuration>
    Configuration force(Configuration & config, Configuration & force){
        force.zero();
		
		recursive_sum_force<0>(config, force);

        return force;
    }

	template<class Configuration>
    Configuration grad(Configuration & config, Configuration & grad){
        grad.zero();
		
		recursive_sum_grad<0>(config, grad);

        return grad;
    }

};

} // namespace NSL::Action

#include "Implementations/hubbardFermiAction.hpp"
#include "Implementations/hubbardGaugeAction.hpp"

#endif //NANOSYSTEMLIBRARY_ACTION_HPP*//*