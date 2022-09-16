#ifndef NANOSYSTEMLIBRARY_HUBBARDGAUGEACTION_HPP
#define NANOSYSTEMLIBRARY_HUBBARDGAUGEACTION_HPP

#include "../action.tpp"

namespace NSL::Action {

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
class HubbardGaugeAction;

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
struct params<HubbardGaugeAction<Type, TensorType>> { 
	float U = 0.;
	float beta = 1.;
	int nt = 16;
};

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
class HubbardGaugeAction : public BaseAction<Type, TensorType> {
	private:
	params<HubbardGaugeAction> par;
	double Utilde;

	public:
	typedef Type type;
	HubbardGaugeAction(params<HubbardGaugeAction> _params) : par(_params), Utilde(par.U * par.beta / par.nt){}
	Configuration<type> force(const Tensor<TensorType>& phi);
	Configuration<type> grad(const Tensor<TensorType>& phi);
	type eval(const Tensor<TensorType>& field);


};
}

#endif // NANOSYSTEMLIBRARY_HUBBARDGAUGEACTION_HPP