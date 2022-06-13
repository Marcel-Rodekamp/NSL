#ifndef NANOSYSTEMLIBRARY_HUBBARDGAUGEACTION_HPP
#define NANOSYSTEMLIBRARY_HUBBARDGAUGEACTION_HPP

#include "../action.tpp"

namespace NSL::Action {

class HubbardGaugeAction;

template<> 
struct params<HubbardGaugeAction> { 
	double U=0.; 
};

class HubbardGaugeAction : public BaseAction<complex<double>> {
	private:
	double U;

	public:
	HubbardGaugeAction(params<HubbardGaugeAction> params) : U(params.U){}
	Configuration<complex<double>> force(const Tensor<complex<double>>& phi){return Configuration<complex<double>>({"force", phi});};
	Configuration<complex<double>> grad(const Tensor<complex<double>>& phi){return Configuration<complex<double>>({"grad", phi});};
	complex<double> eval(const Tensor<complex<double>>& field) { return field[1]*U; };
};

}

#endif // NANOSYSTEMLIBRARY_HUBBARDGAUGEACTION_HPP