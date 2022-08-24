//! \file example_configuration.cpp
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
	NSL::Tensor<NSL::complex<double>> phi1(2,2); phi1.rand(); 
	NSL::Tensor<NSL::complex<double>> phi2(2,2); phi2.rand(); 
    // NSL::Configuration
	NSL::Configuration<NSL::complex<double>> config{
		{"phi",phi1}, 
        {"phi2",phi2} 
	};

	NSL::Tensor<NSL::complex<double>> force1(phi1, true);
	NSL::Tensor<NSL::complex<double>> force2(phi2, true);
	NSL::Tensor<NSL::complex<double>> grad1(phi1, true);
	NSL::Tensor<NSL::complex<double>> grad2(phi2, true);

	std::cout << phi1 << std::endl;
    std::cout << phi2 << std::endl;

	NSL::Action::HubbardFermiAction act1 = NSL::Action::HubbardFermiAction({1.0, 1.0, 1.0});
	NSL::Action::HubbardGaugeAction act2 = NSL::Action::HubbardGaugeAction({42.0});
	
	auto act3 = NSL::Action::SingleAction<NSL::Action::HubbardFermiAction>("phi", {1.0, 1.0, 1.0});
	auto act4 = NSL::Action::SingleAction<NSL::Action::HubbardGaugeAction>("phi", {42.0});
	
	// NSL::Action::Action act5(act3, act4);
	NSL::Action::Action act5 = act3;
	NSL::Action::Action act6 = act3 + act4;

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

