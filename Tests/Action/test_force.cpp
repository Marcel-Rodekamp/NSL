#include "../test.hpp"
#include <highfive/H5File.hpp>
#include <iostream>

template<typename Type,template<typename Type_, typename Lattice_> class HFM >
void test_force();

// =============================================================================
// Test Cases
// =============================================================================

COMPLEX_NSL_TEST_CASE( "Action: tests force routine and compares to finite differencing", "[Action,force,default]" ) {
    INFO("Hubbard Exponential");
    test_force<TestType,NSL::FermionMatrix::HubbardExp>();
    INFO("Hubbard Diagonal");
    test_force<TestType,NSL::FermionMatrix::HubbardDiag>();
}

//=======================================================================
// Implementation Details
//=======================================================================

template<typename Type,template<typename Type_, typename Lattice_> class HFM >
void test_force(){

    // Typically you want to read these in or provide as an argument to such 
    // a code but for this example we just specify them here
    // System Parameters
    //    Inverse temperature 
    Type beta = 2.;
    //    On-Site Coupling
    Type U    = 3.0;
    //    Number of time slices
    NSL::size_t Nt =8;
    //    Number of ions (spatial sites)
    NSL::size_t Nx =  2;

    // Define the lattice geometry of interest
    // ToDo: Required for more sophisticated actions
    NSL::Lattice::Ring<Type> lattice(Nx); 

    NSL::Logger::info("Setting up a Hubbard action with beta={}, Nt={}, U={}, on a ring with {} sites.", NSL::real(beta), Nt, NSL::real(U), Nx);

    // Put the action parameters into the appropriate container
    typename NSL::Action::HubbardGaugeAction<Type>::Parameters params(
        /*beta=*/  beta,
        /*Nt = */  Nt,    
        /*U =  */  U
    );
    typename NSL::Action::HubbardFermionAction<Type,decltype(lattice),
        HFM<Type,decltype(lattice)>>::Parameters paramsHFM(
        /*beta=*/  beta,
        /*Nt = */  Nt,    
        /*lattice=*/lattice
    );

    NSL::Action::HubbardGaugeAction<Type> S_gauge(params);
    NSL::Action::HubbardFermionAction<Type,decltype(lattice),HFM<Type,decltype(lattice)>> S_fermion(paramsHFM);

    // Initialize the action
    NSL::Action::Action S = S_gauge + S_fermion;

    // Initialize a configuration as starting point for the MC change
    // For CPU code put here
    NSL::Configuration<Type> config{
        {"phi", NSL::Tensor<Type>(Nt,Nx)}
    };
    config["phi"].randn();
    config["phi"].imag() = 0; // use purely real fields
    
    config["phi"] *= params.Utilde;

    NSL::Configuration<Type> gradS{
        {"phi", NSL::Tensor<Type>(Nt,Nx)}
    };
    // This is how one computes the grad of the action
    gradS = S.grad(config);

    REQUIRE( (config["phi"].imag() == S.force(config)["phi"].imag() ).all() );  // the force should all real (when chemical potential is zero) 
    
    std::cout << std::endl;
    NSL::RealTypeOf<Type> epsilon = 0.0001;
    auto S_val = S(config);
    for (int t=0; t< Nt; t++){
        for (int i=0;i<Nx;i++) {
            NSL::Configuration<Type> configE(config,true);
	        configE["phi"](t,i) += epsilon;
            
            auto fin_diff = (S(configE)-S_val)/epsilon;
            auto err = NSL::real(NSL::LinAlg::abs(fin_diff-gradS["phi"](t,i)));
            int fin_diff_order = getMatchingDigits(err);
            int eps_order = getMatchingDigits(epsilon);

            std::string repr = fmt::format("t={},x={}",t,i);
            INFO( repr );
            repr = fmt::format("ΔS = {}",NSL::to_string(fin_diff));
            INFO( repr );
            repr = fmt::format("∂S = {}",NSL::to_string(gradS["phi"](t,i)));
            INFO( repr );
            repr = fmt::format("err= {}",err);
            INFO( repr );
            repr = fmt::format(" ε = {}",epsilon);
            REQUIRE( fin_diff_order >= eps_order );

	        //  Note! this test will ALWAYS fail for complex<float>.  The logDetM routine is too imprecise in this case when calculating the finite differencing!
        }
    }
}




