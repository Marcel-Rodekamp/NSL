#include "../test.hpp"
#include <highfive/H5File.hpp>
#include <iostream>

template<typename Type>
void test_force();

// =============================================================================
// Test Cases
// =============================================================================

COMPLEX_NSL_TEST_CASE( "Action: tests force routine and compares to finite differencing", "[Action,force,default]" ) {
  test_force<TestType>();
}

//=======================================================================
// Implementation Details
//=======================================================================

template<typename Type>
void test_force(){

  //  typedef NSL::complex<double> Type;

    // Typically you want to read these in or provide as an argument to such 
    // a code but for this example we just specify them here
    // System Parameters
    //    Inverse temperature 
    Type beta = 1.;
    //    On-Site Coupling
    Type U    = 2.0;
    //    Number of time slices
    NSL::size_t Nt =4;
    //    Number of ions (spatial sites)
    NSL::size_t Nx =  2;

    // Define the lattice geometry of interest
    // ToDo: Required for more sophisticated actions
    NSL::Lattice::Ring<Type> lattice(Nx); 

    NSL::Logger::info("Setting up a Hubbard-Gauge action with beta={}, Nt={}, U={}, on a ring with {} sites.", NSL::real(beta), Nt, NSL::real(U), Nx);

    // Put the action parameters into the appropriate container
    typename NSL::Action::HubbardGaugeAction<Type>::Parameters params(
        /*beta=*/  beta,
        /*Nt = */  Nt,    
        /*U =  */  U
    );
    typename NSL::Action::HubbardFermionAction<Type,decltype(lattice),
        NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>>::Parameters paramsHFM(
        /*beta=*/  beta,
        /*Nt = */  Nt,    
        /*lattice=*/lattice
    );

    NSL::Action::HubbardGaugeAction<Type> S_gauge(params);
    NSL::Action::HubbardFermionAction<Type,decltype(lattice),NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>> S_fermi(paramsHFM);

    // Initialize the action
    NSL::Action::Action S = S_gauge + S_fermi;

    // Initialize a configuration as starting point for the MC change
    // For CPU code put here
    NSL::Configuration<Type> config{
        {"phi", NSL::Tensor<Type>(Nt,Nx)}
    };
    config["phi"].rand();

  // Compute the action
    std::cout << "Actions -> eval (Configurations)" << std::endl;
    std::cout << S(config) << std::endl;
  // or use (the operator() just calls this function)
  //std::cout << S.eval(config) << std::endl;

  // Compute the force
    std::cout << "Actions -> force" << std::endl;
    std::cout << S.force(config) << std::endl;

  // Compute the gradient dS/dPhi
  //  std::cout << "Actions -> grad" << std::endl;
  //  std::cout << S.grad(config) << std::endl;
	
  //  REQUIRE( (pin == pout).all() );
}




