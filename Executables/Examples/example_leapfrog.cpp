#include "Action/Implementations/hubbardGaugeAction.tpp"
#include "Integrator/Impl/leapfrog.tpp"
#include "NSL.hpp"

int main(){
    
    typedef NSL::complex<double> cd;
	
    NSL::Tensor<cd> phi(2,2); phi.rand(); 
    NSL::Tensor<cd> pi(2,2); pi.rand(); 

    // define configuration
	NSL::Configuration<cd> config{
		{"phi",phi}, 
	};

    // define momentum
    NSL::Configuration<cd> momentum{
		{"phi",pi}, 
	};

    // define the parameters for the action
    NSL::Action::HubbardGaugeAction<cd>::Parameters params(
        /*beta=*/  1,
        /*Nt = */  32,    
        /*U =  */  1
    );

    // define the action
	NSL::Action::Action S = NSL::Action::HubbardGaugeAction<cd>(params);

    // define integrator
    NSL::Integrator::Leapfrog LF(
        /*action=*/ S,
        /*trajectoryLength=*/ 1,
        /*numberSteps=*/ 10,
        /*backward*/ false // optional
    );

    // integrate eom
    auto [config_proposal,momentum_proposal] = LF(/*q=*/config,/*p*/ momentum);

    std::cout << config["phi"] << std::endl;
    std::cout << momentum["phi"] << std::endl;

    std::cout << config_proposal["phi"] << std::endl;
    std::cout << momentum_proposal["phi"] << std::endl;

    return EXIT_SUCCESS;
}
