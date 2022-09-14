#ifndef NANOSYSTEMLIBRARY_HUBBARDGAUGEACTION_HPP
#define NANOSYSTEMLIBRARY_HUBBARDGAUGEACTION_HPP

#include "../action.tpp"

namespace NSL::Action {

class HubbardGaugeAction;

template<> 
struct params<HubbardGaugeAction> { 
	double U = 0.;
	double beta = 1.;
	int nt = 16;
};

class HubbardGaugeAction : public BaseAction<complex<double>> {
	public:
	HubbardGaugeAction(params<HubbardGaugeAction> _params) : par(_params), Utilde(par.U * par.beta / par.nt){}
	Configuration<complex<double>> force(const Tensor<complex<double>>& phi);
	Configuration<complex<double>> grad(const Tensor<complex<double>>& phi);
	complex<double> eval(const Tensor<complex<double>>& field);

	private:
	params<HubbardGaugeAction> par;
	double Utilde;

};
}

#endif // NANOSYSTEMLIBRARY_HUBBARDGAUGEACTION_HPP