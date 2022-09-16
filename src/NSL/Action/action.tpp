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

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber ... TensorTypes>
class BaseAction{
	public:
	typedef Type type;
	virtual Configuration<TensorTypes...> force(const Tensor<TensorTypes>&... fields) = 0;
	virtual Configuration<TensorTypes...> grad(const Tensor<TensorTypes>&... fields) = 0;
	virtual Type eval(const Tensor<TensorTypes>& ... fields) = 0;
};

/*! A wrapper class for Action implementations.
 * Handles the interaction between configurations and actions
 **/

template<class ActionImp> 
class SingleAction {
	private:
	ActionImp Act;
	std::string key;		//TODO make general for actions with multiple fields... How would ActionImp return the force of multiple fields?
	
	public:
	typedef ActionImp::type type;
	SingleAction(std::string pkey, params<ActionImp> params) : Act(ActionImp(params)), key(pkey) {}

	template<class Configuration>
	type eval(Configuration& config){ 
		return Act.eval(config[key]); 
	};

	template<class Configuration>
	Configuration force(Configuration & config){
		return Configuration{{key, Act.force(config[key])["force"]}};
	};

	template<class Configuration>
	Configuration grad(Configuration & config){
		return Configuration({{key, Act.grad(config[key])["grad"]}});
	};

};

/*! Container class for modular actions.
 * Offers the same functionality as the Action class.
 * It contains multiple actions and sums up their outputs.
 **/

template<class ...SingleActions>
class Action {	
public:
	std::tuple<SingleActions...>  Summands;
	typedef std::tuple_element<0, std::tuple<SingleActions...>>::type type;				//TODO deduce type of sum
	Action(SingleActions ... pSummands) :Summands(pSummands ...){}

	// template<class SingleAction>
	// Action<SingleActions...,SingleAction> operator += (const SingleAction& other){
	// 	return *this + other;
	// }

	template<class Configuration>
    type eval(Configuration & config){
		type sum = 0;
        recursive_sum_eval<0>(sum, config);
        return sum;
    }
	template<class Configuration>
    Configuration force(Configuration & config){
		Configuration sum;
		recursive_sum_force<0>(sum, config);
        return sum;
    }
	template<class Configuration>
    Configuration grad(Configuration & config){
		Configuration sum;
		recursive_sum_grad<0>(sum, config);

        return sum;
    }

	private:

	template<int I, class Configuration>
    void recursive_sum_eval(type & sum, Configuration & config){
        if constexpr (I < sizeof...(SingleActions)){
            sum += std::get<I>(Summands).eval(config);
            recursive_sum_eval<I+1>(sum, config);
        } 
    }

	template<int I, class Configuration>
    void recursive_sum_force(Configuration & sum, Configuration & config){
        if constexpr (I < sizeof...(SingleActions)){
			sum += std::get<I>(Summands).force(config);
			recursive_sum_force<I+1>(sum, config);
        } 
    }

	template<int I, class Configuration>
    void recursive_sum_grad(Configuration & sum, Configuration & config){
        if constexpr (I < sizeof...(SingleActions)){
			sum += std::get<I>(Summands).grad(config);
			recursive_sum_grad<I+1>(sum, config);
        } 
    }
};

template<class ...SingleActions1, class ...SingleActions2>
Action<SingleActions1... , SingleActions2...> operator+ ( const Action<SingleActions1...> & left, const Action<SingleActions2...> & right ){
	auto Summands = std::tuple_cat(left.Summands, right.Summands);
	return std::make_from_tuple<Action<SingleActions1... , SingleActions2...>>(Summands);
}

template<class ...SingleActions1, class SingleAction2>
Action<SingleActions1... , SingleAction2> operator+ ( const Action<SingleActions1...> & left, const SingleAction2 & right ){
	auto Summands = std::tuple_cat(left.Summands, std::make_tuple(right));
	return std::make_from_tuple<Action<SingleActions1... , SingleAction2>>(Summands);
}

template<class SingleAction1, class ...SingleActions2>
Action<SingleAction1 , SingleActions2...> operator+ ( const SingleAction1 & left, const Action<SingleActions2...> & right ){
	auto Summands = std::tuple_cat(std::make_tuple(left), right.Summands);
	return std::make_from_tuple<Action<SingleAction1 , SingleActions2...>>(Summands);
}

template<class SingleAction1, class SingleAction2>
Action<SingleAction1 , SingleAction2> operator+ ( const SingleAction1 & left, const SingleAction2 & right ){
	auto Summands = std::make_tuple(left, right);
	return std::make_from_tuple<Action<SingleAction1 , SingleAction2>>(Summands);
};

} // namespace NSL::Action

#include "Implementations/hubbardGaugeAction.cpp"

#endif //NANOSYSTEMLIBRARY_ACTION_HPP*//*