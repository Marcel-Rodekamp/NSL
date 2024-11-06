#include "Action/Implementations/hubbardGaugeAction.tpp"
#include "Action/Implementations/hubbardFermiAction.tpp"
#include "Integrator/Impl/leapfrog.tpp"
#include "NSL.hpp"

int main(int argc, char** argv){
    
    typedef NSL::complex<double> cd;
	
    NSL::size_t Nx =  8;
    NSL::size_t Nt =  32;
    NSL::Tensor<cd> phi(Nt,Nx); phi.randn(); 
    NSL::Tensor<cd> pi(Nt,Nx); pi.randn();
    NSL::Lattice::Ring<cd> lattice(Nx); 


    NSL::Parameter params2 = NSL::init(argc, argv, "Hubbard Model HMC with Pseudofermions");
    
    phi.imag() = 0;
    pi.imag() = 0;
   
    // define configuration
    NSL::Configuration<cd> config{
		{"phi",phi}, 
    };

    NSL::Parameter params;
    params["beta"] = 10.;
    params["Nt"] = Nt;
    params["U"] = 3.;
    params["mu"] = 0.;


    // define momentum
    NSL::Configuration<cd> momentum{
		{"phi",pi}, 
	};

    NSL::Action::HubbardGaugeAction<cd> S_gauge(params);
    NSL::Action::PseudoFermionAction<cd,decltype(lattice), NSL::FermionMatrix::HubbardExp<cd,decltype(lattice)>>S_fermion(lattice, params);
    //NSL::Action::HubbardFermionAction<cd,decltype(lattice),NSL::FermionMatrix::HubbardExp<cd,decltype(lattice)>> S_fermion(lattice, params);
    // define the action
    NSL::Action::Action S = S_gauge + S_fermion;

    // compute pseudo fermions (if they don't exist this call does nothing)
    S.computePseudoFermion(config);
    
    cd Hi, Hf;

    Hi = (momentum["phi"] * momentum["phi"]).sum()/2.0 + S(config);

    for (int Nmd = 10; Nmd < 210; Nmd += 10){
      // define integrator
      NSL::Integrator::Leapfrog LF(
        /*action=*/ S,
        /*trajectoryLength=*/ 1,
        /*numberSteps=*/ Nmd,
        /*backward*/ false // optional
      );

      // integrate eom
      auto [config_proposal,momentum_proposal] = LF(/*q=*/config,/*p*/ momentum);
 
      Hf = (momentum_proposal["phi"] * momentum_proposal["phi"]).sum()/2.0 + S(config_proposal);
      std::cout << Nmd << "\t" << NSL::LinAlg::abs((Hf-Hi).real()/Hi.real()) << std::endl;
    }

    return EXIT_SUCCESS;
}
