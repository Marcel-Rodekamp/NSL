#ifndef NSL_PSEUDO_FERMION_ACTION_TPP
#define NSL_PSEUDO_FERMION_ACTION_TPP

namespace NSL::Action {

template<NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType 
>
class PseudoFermionAction: public BaseAction<Type,Type>{
    public:
        
        PseudoFermionAction(LatticeType & lattice, NSL::Parameter & params) :
            BaseAction<Type,Type>("phi"),
            FM_(lattice, params),
            chi_(),
            pseudoFermion_()
        {}

        PseudoFermionAction(LatticeType & lattice, NSL::Parameter & params, const std::string & fieldName) :
            BaseAction<Type,Type>(fieldName),
            FM_(lattice, params),
            chi_(),
            pseudoFermion_()
        {}

    // We import the eval/grad/force functions from the BaseAction such 
    // that we do not need to reimplement the Configuration based versions
    // We don't understand why this is not automatically done, probably due 
    // to BaseAction being an abstract base class
    using BaseAction<Type,Type>::eval;
    using BaseAction<Type,Type>::grad;
    using BaseAction<Type,Type>::force;

	Configuration<Type> force(const Tensor<Type>& phi);
	Configuration<Type> grad(const Tensor<Type>& phi);
	Type eval(const Tensor<Type>& phi);

    NSL::Configuration<Type> pseudoFermion(){
        return {{
            this->configKey_,
            pseudoFermion_
        }};
    }

    void pseudoFermion(const NSL::Tensor<Type> & pf){
        pseudoFermion_ = pf;
    }

    //! compute the pseudofermion 
    /*! 
     * This method assumes that the vector \f$\chi\f$ has been sampled already.
     * */
    bool computePseudoFermion(const NSL::Configuration<Type> & config) {
        // sqrt(0.5) = 0.707... is used to remove the factor 1/2 from the 
        // normal distribution. chi_ ~ exp(-Chi^+ Chi)
        chi_ = NSL::randn_like(config.at(this->configKey_), 0., 0.7071067811865476 );
        //chi_ = NSL::randn_like(config.at(this->configKey_));

        // populate the fermion matrix 
        FM_.populate(config.at(this->configKey_));

        // compute the pseudo fermion
        // chi_ has been sampled already
        pseudoFermion_ = FM_.M( chi_ );

        return true;
    }

    protected:
        FermionMatrixType FM_;
        //NSL::LinAlg::CG<Type> cg_;

        NSL::Tensor<Type> chi_;
        NSL::Tensor<Type> pseudoFermion_;
};

template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType 
>
Type PseudoFermionAction<Type,LatticeType,FermionMatrixType>::eval(const Tensor<Type>& phi){
    // compute pseudo fermion; This sets the tensor pseudoFermion_
    // and populates the fermion matrix
    FM_.populate(phi);

    NSL::LinAlg::CG<Type> cg_(FM_, NSL::FermionMatrix::MMdagger);

    // compute MMdagger * pseudoFermion
    NSL::Tensor<Type> MMdaggerInv = cg_(pseudoFermion_);
    
    // The pseudo fermion action is then given by the inner product
    return NSL::LinAlg::inner_product(pseudoFermion_,MMdaggerInv);
}
	
template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
Configuration<Type> PseudoFermionAction<Type,LatticeType,FermionMatrixType>::force(const Tensor<Type>& phi){
    // The force has an explicit minus sign from the gradient
    return (-1)*this->grad(phi);
}

template<
    NSL::Concept::isNumber Type, 
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
Configuration<Type> PseudoFermionAction<Type,LatticeType,FermionMatrixType>::grad(const Tensor<Type>& phi){
    
    // Compute pseudo fermion (this populates the fermion matrix)
    FM_.populate(phi);

    // calculate (MM^+)^{-1} * pseudoFermion
    NSL::LinAlg::CG<Type> cg_(FM_, NSL::FermionMatrix::MMdagger);

    NSL::Tensor<Type> MMdaggerInv = cg_(pseudoFermion_);

    // calculate the derivatives of the fermion matrix
    return NSL::Configuration<Type> {{ this->configKey_,
        -2.*FM_.dMdPhi(
            /*left*/NSL::LinAlg::conj(MMdaggerInv),/*right*/FM_.Mdagger(MMdaggerInv)
        ).real()
        // -FM_.dMdaggerdPhi(
        //     /*left*/NSL::LinAlg::conj(FM_.Mdagger(MMdaggerInv)),/*right*/MMdaggerInv
        // )
    }};
}


} // namespace NSL::Action

#endif // NSL_PSEUDO_FERMION_ACTION_TPP
