/*!
 * This example shows how a `NSL::Action` is constructed and used.
 * */
#include <iostream>
#include "Action.hpp"

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

	NSL::Action::HubbardGaugeAction<cd, cd> act1({1.0, 1.0, 10});
	NSL::Action::HubbardGaugeAction<cd, cd> act2({2.0, 2.0, 20});

	NSL::Action::SingleAction<NSL::Action::HubbardGaugeAction<cd, cd>> act3("phi", {1.0, 1.0, 10});
	NSL::Action::SingleAction<NSL::Action::HubbardGaugeAction<cd, cd>> act4("phi", {2.0, 2.0, 20});

	// NSL::Action::Action act5(act3, act4);
	NSL::Action::Action act5 = act3 + act4;

	std::cout << "Actions -> eval (TensorTypes)" << std::endl;
	std::cout << act1.eval(phi1) << std::endl;
	std::cout << act2.eval(phi1) << std::endl;

	std::cout << "Actions -> eval (Configurations)" << std::endl;
	std::cout << act3.eval(config) << std::endl;
	std::cout << act4.eval(config) << std::endl;
	std::cout << "Actions -> eval (SumAction)" << std::endl;
	std::cout << act5.eval(config) << std::endl << std::endl;

	std::cout << "Actions -> force (TensorTypes)" << std::endl;
	std::cout << act1.force(phi1) << std::endl;
	std::cout << act2.force(phi1) << std::endl;

	std::cout << "Actions -> force (Configurations)" << std::endl;
	std::cout << act3.force(config) << std::endl;
	std::cout << act4.force(config) << std::endl;

	std::cout << "Actions -> force (SumAction)" << std::endl;
	std::cout << act5.force(config) << std::endl;

	std::cout << "Actions -> grad (TensorTypes)" << std::endl;
	std::cout << act1.grad(phi1) << std::endl;
	std::cout << act2.grad(phi1) << std::endl;

	std::cout << "Actions -> grad (Configurations)" << std::endl;
	std::cout << act3.grad(config) << std::endl;
	std::cout << act4.grad(config) << std::endl;

	std::cout << "Actions -> grad (SumAction)" << std::endl;
	std::cout << act5.grad(config) << std::endl;

	return EXIT_SUCCESS;

}