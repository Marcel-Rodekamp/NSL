#ifndef NSL_HUBBARD_GAUGE_ACTION_TPP
#define NSL_HUBBARD_GAUGE_ACTION_TPP

#include "../action.tpp"
#include "../actionParams.tpp"

namespace NSL::Action {

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
class HubbardGaugeAction;

//! Parameter set for the Hubbard Gauge Action
/*!
 * This parameters contain 
 * * `U` The on-site interaction
 * * `beta` The inverse temperature
 * * `Nt` The number of time slice (troterization)
 * */
template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
struct Parameters<HubbardGaugeAction<Type,TensorType>> { 
    public:
    // inverse temperature
	const Type beta;

    // time slices
    const NSL::size_t Nt;

    // on-site interaction
	const Type U;

    Parameters(const Type & beta, const NSL::size_t & Nt, const Type & U) :
        beta(beta), Nt(Nt), U(U), delta_(beta/Nt)
    {}

    //! Get the on-site interaction in lattice units.
    const Type Utilde(){
        return delta_*U;
    }

    private:

    // lattice spacing
    const Type delta_;
};

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
class HubbardGaugeAction : 
    public BaseAction<
        Parameters<HubbardGaugeAction<Type,TensorType>>, 
        Type, TensorType
    > 
{

	HubbardGaugeAction(const Parameters<HubbardGaugeAction<Type,TensorType>> & params) : 
        BaseAction<Parameters<HubbardGaugeAction<Type,TensorType>>, Type, TensorType>(
            params, "phi"
        )
    {}

	Configuration<TensorType> force(const Tensor<TensorType>& phi);
	Configuration<TensorType> grad(const Tensor<TensorType>& phi);
	Type eval(const Tensor<TensorType>& field);
};

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
Type HubbardGaugeAction<Type, TensorType>::eval(const Tensor<TensorType>& field){
    return (field * field).sum() / 2 / this->Utilde;
}
	
template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
Configuration<TensorType> HubbardGaugeAction<Type, TensorType>::force(const Tensor<TensorType>& phi){
    return Configuration<Type>{{"force", phi /- this->params_.Utilde()}};
}

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
Configuration<TensorType> HubbardGaugeAction<Type, TensorType>::grad(const Tensor<TensorType>& phi){
    return Configuration<Type>{{"grad", phi / this->params_.Utilde()}};
}

} // namespace NSL::Action

#endif // NSL_HUBBARD_GAUGE_ACTION_TPP
