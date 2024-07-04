#ifndef NSL_WILSON_FERMION_ACTION_TPP
#define NSL_WILSON_FERMION_ACTION_TPP



namespace NSL::Action {

template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType, 
    NSL::Concept::isNumber TensorType = Type
>
class WilsonFermionAction : 
    public BaseAction<Type, TensorType> 
{   
    public: 

	WilsonFermionAction(LatticeType & lattice, NSL::Parameter & params, const std::string & fieldName) : 
        BaseAction<Type, TensorType>(fieldName),
        params_(params),
        wfm_(lattice,params),
        Nf(params["Nf"].to<NSL::size_t>())
    {}

	WilsonFermionAction(LatticeType & lattice, NSL::Parameter & params): 
	    WilsonFermionAction<Type,LatticeType,FermionMatrixType>(lattice,params,"U")
    {}

    // We import the eval/grad/force functions from the BaseAction such 
    // that we do not need to reimplement the Configuration based versions
    // We don't understand why this is not automatically done, probably due 
    // to BaseAction being an abstract base class
    using BaseAction<Type,TensorType>::eval;
    using BaseAction<Type,TensorType>::grad;
    using BaseAction<Type,TensorType>::force;

	Configuration<TensorType> force(const Tensor<TensorType>& phi);
	Configuration<TensorType> grad(const Tensor<TensorType>& phi);
	Type eval(const Tensor<TensorType>& phi);

    protected:
    NSL::Parameter params_;
    NSL::size_t Nf;

    FermionMatrixType wfm_;
}; // class WilsonFermiAction

template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType, 
    NSL::Concept::isNumber TensorType
>
Type WilsonFermionAction<Type,LatticeType,FermionMatrixType,TensorType>::eval(const Tensor<TensorType>& U){

    wfm_.populate(U);
    Type logDetM= wfm_.logDetM();

    return (-1)*logDetM*Nf;
}
	
template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType, 
    NSL::Concept::isNumber TensorType
>
Configuration<TensorType> WilsonFermionAction<Type,LatticeType,FermionMatrixType,TensorType>::force(const Tensor<TensorType>& U){
    // The force has an explicit minus sign from the gradient
    return (-1)*this->grad(U);
}

template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType, 
    NSL::Concept::isNumber TensorType
>
Configuration<TensorType> WilsonFermionAction<Type,LatticeType,FermionMatrixType,TensorType>::grad(const Tensor<TensorType>& U){

    NSL::Configuration<TensorType> dS{{ this->configKey_, NSL::zeros_like(U) }};

   
    wfm_.populate(U);
    dS[this->configKey_]+= wfm_.gradLogDetM()*Nf;
    return dS;
}

} // namespace NSL::Action

#endif // NSL_WILSON_FERMION_ACTION_TPP
