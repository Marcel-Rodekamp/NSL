#ifndef NANOSYSTEMLIBRARY_HUBBARDFERMIACTION_HPP
#define NANOSYSTEMLIBRARY_HUBBARDFERMIACTION_HPP

#include "../action.tpp"

namespace NSL::Action {

class HubbardFermiAction;

template<> 
struct params<HubbardFermiAction> {
	// NSL::Lattice::SpatialLattice<typename NSL::RT_extractor<Type>::value_type> lattice;
	double kappa = 1.;
	double beta=1.;
	double mu=0.;
};

class HubbardFermiAction : public BaseAction<complex<double>> {
	private:
	double kappa;
	double beta;
	double mu;

	public:
	HubbardFermiAction(params<HubbardFermiAction> params) : kappa(params.kappa), beta(params.beta), mu(params.mu){}
	Configuration<complex<double>> force(const Tensor<complex<double>>& phi){return Configuration<complex<double>>{{"force", phi}};};
	Configuration<complex<double>> grad(const Tensor<complex<double>>& phi){return Configuration<complex<double>>{{"grad", phi}};};
	complex<double> eval(const Tensor<complex<double>>& field) { return field[0]+mu; };
};


}

#endif // NANOSYSTEMLIBRARY_HUBBARDFERMIACTION_HPP