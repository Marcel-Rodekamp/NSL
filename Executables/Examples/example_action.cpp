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

    NSL::Action::HubbardGaugeAction<cd>::Parameters params(
        /*beta=*/  1,
        /*Nt = */  32,    
        /*U =  */  1
    );

    // create action with just a params class
	NSL::Action::HubbardGaugeAction<cd> act1(params);

    // create the action with an auto induced parameter class 
    // The order is always {beta,Nt,U}
    // Provide a specialized field name as second argument
	NSL::Action::HubbardGaugeAction<cd> act2({2, 16, 1}, "phi2");

    // Add the two actions to form the final desired action
	// NSL::Action::Action act5(act3, act4);
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
