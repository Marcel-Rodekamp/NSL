#include "hubbardGaugeAction.hpp"

namespace NSL::Action {

	template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
	Type HubbardGaugeAction<Type, TensorType>::eval(const Tensor<TensorType>& field){
		return (field * field).sum() / 2 / this->Utilde;
	}
	
	template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
	Configuration<Type> HubbardGaugeAction<Type, TensorType>::force(const Tensor<TensorType>& phi){
		return Configuration<Type>{{"force", phi /-this->Utilde}};
	}

	template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
	Configuration<Type> HubbardGaugeAction<Type, TensorType>::grad(const Tensor<TensorType>& phi){
		return Configuration<Type>{{"grad", phi / this->Utilde}};
	}

}  // namespace HubbardGaugeAction
