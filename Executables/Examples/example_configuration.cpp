//! \file example_configuration.cpp
/*!
 * This example shows how a `NSL::Configuration` is constructed and used.
 * */

#include <iostream>
#include "Configuration.hpp"

int main(){
    // Examples:
    // 1. Non-Homogeneous Configuration
    NSL::Tensor<int> spins(8,4);spins.randint(2);
    NSL::Tensor<double> phase(4);phase.rand();
    NSL::Tensor<NSL::complex<double>> phi(8,4); phi.rand(); 

    
    NSL::Configuration<int,double,NSL::complex<double>> config( 
        {"spin",spins}, 
        {"U",phase}, 
        {"phi",phi} 
    );

    // Compilation error for non-homogeneus configurations:
    // std::cout << "spin: " << config.field("spin") << std::endl;
    std::cout << "spin: " << config.field<int>("spin") << std::endl;
    std::cout << "U   : " << config.field<double>("U") << std::endl;
    std::cout << "phi : " << config.field<NSL::complex<double>>("phi") << std::endl;

    // Access all keys:
    std::cout << "[ ";
    for(auto fName: config.fieldNames()){
        std::cout << fName << " ";
    }
    std::cout << "]" << std::endl;

    // 2. Homogeneous Configuration


    NSL::Tensor<int> spins2(8,2);spins2.randint(2);

    NSL::Configuration<int,int> homConfig( 
        {"spin 1",spins}, 
        {"spin 2",spins2} 
    );

   // for homogeneous configs it works without template parameter
   std::cout << "spin 1 = " << homConfig.field("spin 1") << std::endl;
   std::cout << "spin 2 = " << homConfig.field("spin 2") << std::endl;


    return EXIT_SUCCESS;

}

