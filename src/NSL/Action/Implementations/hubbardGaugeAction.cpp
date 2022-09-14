#include "hubbardGaugeAction.hpp"

namespace NSL::Action {

	complex<double> HubbardGaugeAction::eval(const Tensor<complex<double>>& field){
		return (field * field).sum() / 2 / this->Utilde;
	}
	
	Configuration<complex<double>> HubbardGaugeAction::force(const Tensor<complex<double>>& phi){
		return Configuration<complex<double>>{{"force", phi /-this->Utilde}};
	}

	Configuration<complex<double>> HubbardGaugeAction::grad(const Tensor<complex<double>>& phi){
		return Configuration<complex<double>>{{"grad", phi / this->Utilde}};
	}

}  // namespace HubbardGaugeAction
