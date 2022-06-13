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
    
    NSL::Configuration<NSL::complex<double>,NSL::complex<double>> config( 
        {"phi1",phi1}, 
        {"phi2",phi2} 
    );

	NSL::Tensor<NSL::complex<double>> force1(phi1, true);
	NSL::Tensor<NSL::complex<double>> force2(phi2, true);
	NSL::Configuration<NSL::complex<double>, NSL::complex<double>> force({"phi1", force1}, {"phi2", force2});
	NSL::Tensor<NSL::complex<double>> grad1(phi1, true);
	NSL::Tensor<NSL::complex<double>> grad2(phi2, true);
	NSL::Configuration<NSL::complex<double>, NSL::complex<double>> grad({"phi1", grad1}, {"phi2", grad2});

	std::cout << "phi1 : " << config.field<NSL::complex<double>>("phi1") << std::endl << std::endl;
    std::cout << "phi2 : " << config.field<NSL::complex<double>>("phi2") << std::endl << std::endl;

	NSL::Action::HubbardFermiAction act1 = NSL::Action::HubbardFermiAction({1.0, 1.0, 1.0});
	NSL::Action::HubbardGaugeAction act2 = NSL::Action::HubbardGaugeAction({42.0});
	
	NSL::Action::Action<NSL::Action::HubbardFermiAction> act3 = NSL::Action::Action<NSL::Action::HubbardFermiAction>("phi1", {1.0, 1.0, 1.0});
	NSL::Action::Action<NSL::Action::HubbardGaugeAction> act4 = NSL::Action::Action<NSL::Action::HubbardGaugeAction>("phi2", {42.0});
	
	auto act5 = NSL::Action::SumAction<NSL::Action::Action<NSL::Action::HubbardFermiAction>, NSL::Action::Action<NSL::Action::HubbardGaugeAction>>(act3, act4);
	
	std::cout << "Actions -> eval (TensorTypes)" << std::endl;
	std::cout << act1.eval(phi1) << std::endl;
	std::cout << act2.eval(phi2) << std::endl;
	
	std::cout << "Actions -> eval (Configurations)" << std::endl;
	std::cout << act3.eval(config) << std::endl;
	std::cout << act4.eval(config) << std::endl;

	std::cout << "Actions -> eval (SumAction)" << std::endl;
	std::cout << act5.eval(config) << std::endl << std::endl;

	std::cout << "Actions -> force (TensorTypes)" << std::endl;
	std::cout << act1.force(phi1).field("force") << std::endl;
	std::cout << act2.force(phi2).field("force") << std::endl;
	
	std::cout << "Actions -> force (Configurations)" << std::endl;
	std::cout << act3.force(config, force).field("phi1") << std::endl;
	std::cout << act3.force(config, force).field("phi2") << std::endl;
	std::cout << act4.force(config, force).field("phi1") << std::endl;
	std::cout << act4.force(config, force).field("phi2") << std::endl;

	std::cout << "Actions -> force (SumAction)" << std::endl;
	std::cout << act5.force(config, force).field("phi1") << std::endl;
	std::cout << act5.force(config, force).field("phi2") << std::endl << std::endl;

	std::cout << "Actions -> grad (TensorTypes)" << std::endl;
	std::cout << act1.grad(phi1).field("grad") << std::endl;
	std::cout << act2.grad(phi2).field("grad") << std::endl;
	
	std::cout << "Actions -> grad (Configurations)" << std::endl;
	std::cout << act3.grad(config, grad).field("phi1") << std::endl;
	std::cout << act3.grad(config, grad).field("phi2") << std::endl;
	std::cout << act4.grad(config, grad).field("phi1") << std::endl;
	std::cout << act4.grad(config, grad).field("phi2") << std::endl;

	std::cout << "Actions -> grad (SumAction)" << std::endl;
	std::cout << act5.grad(config, grad).field("phi1") << std::endl;
	std::cout << act5.grad(config, grad).field("phi2") << std::endl << std::endl;

	return EXIT_SUCCESS;

}

