#ifndef NANOSYSTEMLIBRARY_HUBBARDGAUGEACTION_HPP
#define NANOSYSTEMLIBRARY_HUBBARDGAUGEACTION_HPP

#include "../action.tpp"

namespace NSL::Action {

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
class HubbardGaugeAction;

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
struct params<HubbardGaugeAction<Type, TensorType>> { 
	Type U = 0.;
	Type beta = 1.;
	int nt = 16;
};

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
class HubbardGaugeAction : public BaseAction<Type, TensorType> {
	public:
	typedef Type type;
	HubbardGaugeAction(params<HubbardGaugeAction> _params) : par(_params), Utilde(par.U * par.beta / par.nt){}
	Configuration<type> force(const TensorType& phi);
	Configuration<type> grad(const TensorType& phi);
	type eval(const TensorType& field);

	private:
	params<HubbardGaugeAction> par;
	double Utilde;

};
}

#endif // NANOSYSTEMLIBRARY_HUBBARDGAUGEACTION_HPP