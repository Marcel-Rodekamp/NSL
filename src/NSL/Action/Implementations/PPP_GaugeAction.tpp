#ifndef NSL_PPP_GAUGE_ACTION_TPP
#define NSL_PPP_GAUGE_ACTION_TPP

#include "../action.tpp"
#include "../../Lattice.hpp"
#include "hubbard.tpp"
#include "LinAlg.hpp"

namespace NSL::Action {

//! PPP Gauge Action
/*!
 * Given a phi \f(\Phi\f) this action evaluates 
 * \f[ S(\Phi) = \frac{\Phi^2}{\delta U} \f]
 * where 
 *  - \f(\delta = \frac{\beta}{N_t}\f) is the lattice spacing
 *  - \f(\beta\f) is the inverse temperature
 *  - \f(N_t\f) is the number of time slices (troterization)
 *  - \f(U\f) is the on-site interaction of the PPP model Hamiltonian
 * */
template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType = Type>
class PPPGaugeAction : 
    public BaseAction<Type, TensorType> 
{   
    public: 

	PPPGaugeAction(int Nx, NSL::Parameter & params) : 
        BaseAction<Type, TensorType>("phi"),
        params_(params),
        Utilde_(NSL::Hubbard::tilde<Type>(params,"U")),
	Nx_(Nx),
	invVtilde_(Nx_,Nx_)
    {
    
      for (int i=0;i<Nx_;i++) {
      	  for (int j=0;j<Nx_;j++) {
	      invVtilde_(i,j)=Utilde_/(1+Utilde_*((i-j)*(i-j))*t1_*a_/e2_);
	      std::cout << i << " " << j << " " << invVtilde_(i,j) << std::endl;
	  }
      }
      std::cout << " " << std::endl;
      invVtilde_=NSL::LinAlg::mat_inv(invVtilde_);
      for (int i=0;i<Nx_;i++) {
          for (int j=0;j<Nx_;j++) {
              std::cout << i << " " << j << " " << invVtilde_(i,j) << std::endl;
          }
      }
    exit(0);
    }

	PPPGaugeAction(NSL::Parameter & params,const std::string & fieldName) : 
        BaseAction<Type, TensorType>(fieldName),
        params_(params),
        Utilde_(NSL::Hubbard::tilde<Type>(params,"U")) 
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
    int Nx_; // number of spatial sites
    double e2_= 14.3997; // eV-Angstrom
    double t1_ = 2.8; // eV  hopping strength
    double a_ = 1.4; // lattice spacing in Angstrom
    NSL::Parameter params_;
    Type Utilde_;
    NSL::Tensor<Type> invVtilde_;
    
};

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
Type PPPGaugeAction<Type, TensorType>::eval(const Tensor<TensorType>& phi){
     return NSL::LinAlg::inner_product(phi,NSL::LinAlg::mat_mul(invVtilde_, phi))/2.0;
     //return (phi * phi).sum() / ( 2 * Utilde_ ) ;
}
	
template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
Configuration<TensorType> PPPGaugeAction<Type, TensorType>::force(const Tensor<TensorType>& phi){
     return Configuration<Type>{{this->configKey_,-NSL::LinAlg::mat_mul(invVtilde_, phi)}};
     //return Configuration<Type>{{this->configKey_, phi /(- Utilde_)}};
}

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
Configuration<TensorType> PPPGaugeAction<Type, TensorType>::grad(const Tensor<TensorType>& phi){
     return Configuration<Type>{{this->configKey_, NSL::LinAlg::mat_mul(invVtilde_, phi)}};
     //return Configuration<Type>{{this->configKey_, phi / Utilde_}};
}

} // namespace NSL::Action

#endif // NSL_PPP_GAUGE_ACTION_TPP
