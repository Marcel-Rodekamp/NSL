/*!
 * This example shows how a `NSL::Action` is constructed and used.
 * */
#include <iostream>
#include "NSL.hpp"

template<class Configuration>
Configuration zero(Configuration& config){
	for(auto fName: config.fieldNames()){
		config.field(fName) = zeros_like(config.field(fName));
	}
	return config;
}

int main(){
	typedef NSL::complex<double> cd;

	NSL::Tensor<cd> phi1(2,2); phi1.rand(); 
	NSL::Tensor<cd> phi2(2,2); phi2.rand(); 
    // NSL::Configuration
	NSL::Configuration<cd> config{
		{"phi",phi1}, 
        {"phi2",phi2} 
	};

	std::cout << phi1 << std::endl;
    std::cout << phi2 << std::endl;

    // define parameters for action1
    NSL::Parameter params1;
    params1.addParameter<cd>("beta",1);
    params1.addParameter<NSL::size_t>("Nt",32);
    params1.addParameter<cd>("U",1);

    // create action with just a params class
	NSL::Action::HubbardGaugeAction<cd> act1(params1);

    // define parameters for action2
    NSL::Parameter params2;
    params2.addParameter<cd>("beta",1);
    params2.addParameter<NSL::size_t>("Nt",32);
    params2.addParameter<cd>("U",1);

    // create the action with an auto induced parameter class 
    // The order is always {beta,Nt,U}
    // Provide a specialized field name as second argument
	NSL::Action::HubbardGaugeAction<cd> act2(params2, "phi2");

    // Add the two actions to form the final desired action
	// NSL::Action::Action S(act1, act2);
	NSL::Action::Action S = act1 + act2;

    // Compute the action
	std::cout << "Actions -> eval (Configurations)" << std::endl;
    std::cout << S(config) << std::endl;
    // or use (the operator() just calls this function)
    //std::cout << S.eval(config) << std::endl;

    // Compute the force
	std::cout << "Actions -> force" << std::endl;
	std::cout << S.force(config) << std::endl;

    // Compute the gradient dS/dPhi
	std::cout << "Actions -> grad" << std::endl;
	std::cout << S.grad(config) << std::endl;

	return EXIT_SUCCESS;

}
