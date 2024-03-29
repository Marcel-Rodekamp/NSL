#ifndef NSL_ACTION_TPP
#define NSL_ACTION_TPP

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

#include "Configuration.hpp"
#include "concepts.hpp"
#include "typePromotion.hpp"
#include "map.hpp"

#include<tuple>

namespace NSL::Action {

/*! A base class for actions.
 *      Offers the default functionality of actions and 
 *  	is acting as the parent class for the specific actions.
 **/

template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isNumber TensorType
>
class BaseAction{
	public:
	typedef Type ActionValueType;

	virtual Configuration<TensorType> force(const Tensor<TensorType>& fields) = 0;
	virtual Configuration<TensorType> grad(const Tensor<TensorType>& fields) = 0;
	virtual ActionValueType eval(const Tensor<TensorType>& fields) = 0;

	inline Configuration<TensorType> force(Configuration<TensorType> & config){
		return Configuration<TensorType>{{configKey_, force(config[configKey_])[configKey_]}};
	}

	inline Configuration<TensorType> grad(Configuration<TensorType> & config){
		return Configuration<TensorType>({{configKey_, grad(config[configKey_])[configKey_]}});
	}

	inline Type eval(Configuration<TensorType>& config){ 
		return eval(config[configKey_]); 
	}

	BaseAction(const std::string & configKey) : 
        configKey_(configKey) 
    {}

    protected:
    std::string configKey_;
};


/*! Container class for modular actions.
 * Offers the same functionality as the Action class.
 * It contains multiple actions and sums up their outputs.
 **/

// isTemplateDerived requires the same template structure as the base class
// this can not be satisfied in general. We might want to think harder here
// in the future
//template<NSL::Concept::isTemplateDerived<BaseAction> ... SingleActions>
template<typename ... SingleActions>
class Action {	
public:
	Action(SingleActions ... psummands_) : 
        summands_({psummands_ ...})
    {}

    template<NSL::Concept::isNumber TensorType>
	Configuration<TensorType> force(Configuration<TensorType> & config){
        Configuration<TensorType> sum;
        std::apply(
            [&sum, &config](auto & ... terms) {
                (sum += ... += terms.force(config));
            },
            summands_
        );
        return sum;
    };

    template<NSL::Concept::isNumber TensorType>
	Configuration<TensorType> grad(Configuration<TensorType> & config){
        Configuration<TensorType> sum;
        std::apply(
            [&sum, &config](auto & ... terms) {
                (sum += ... += terms.grad(config));
            },
            summands_
        );
        return sum;
    };

    template<NSL::Concept::isNumber TensorType>
	auto eval(Configuration<TensorType> & config){

        if constexpr (sizeof...(SingleActions)!=1){
            typedef CommonTypeOfPack<typename SingleActions::ActionValueType ...> ReturnTypeProposal;
            ReturnTypeProposal sum = static_cast<ReturnTypeProposal>(0);
            std::apply(
                [&sum, &config](auto & ... terms) {
                    (sum += ... += terms.eval(config));
                },
                summands_
            );

            return sum;

        } else {
            typedef TensorType ReturnTypeProposal;
            ReturnTypeProposal sum = static_cast<ReturnTypeProposal>(0);
            std::apply(
                [&sum, &config](auto & ... terms) {
                    (sum += ... += terms.eval(config));
                },
                summands_
            );

            return sum;

        }


    }


    template<NSL::Concept::isNumber TensorType>
	auto operator()(Configuration<TensorType> & config){
        return this->eval(config);
    }

	private:
	std::tuple<SingleActions...>  summands_;

};

template<class ...SingleActions1, class ...SingleActions2>
Action<SingleActions1... , SingleActions2...> operator+ ( const Action<SingleActions1...> & left, const Action<SingleActions2...> & right ){
	auto summands_ = std::tuple_cat(left.summands_, right.summands_);
	return std::make_from_tuple<Action<SingleActions1... , SingleActions2...>>(summands_);
}

template<class ...SingleActions1, class SingleAction2>
Action<SingleActions1... , SingleAction2> operator+ ( const Action<SingleActions1...> & left, const SingleAction2 & right ){
	auto summands_ = std::tuple_cat(left.summands_, std::make_tuple(right));
	return std::make_from_tuple<Action<SingleActions1... , SingleAction2>>(summands_);
}

template<class SingleAction1, class ...SingleActions2>
Action<SingleAction1 , SingleActions2...> operator+ ( const SingleAction1 & left, const Action<SingleActions2...> & right ){
	auto summands_ = std::tuple_cat(std::make_tuple(left), right.summands_);
	return std::make_from_tuple<Action<SingleAction1 , SingleActions2...>>(summands_);
}

template<class SingleAction1, class SingleAction2>
Action<SingleAction1 , SingleAction2> operator+ ( const SingleAction1 & left, const SingleAction2 & right ){
	auto summands_ = std::make_tuple(left, right);
	return std::make_from_tuple<Action<SingleAction1 , SingleAction2>>(summands_);
};

} // namespace NSL::Action

#endif //NSL_ACTION_TPP*//*
