#include <chrono>

#include "Configuration/Configuration.tpp"
#include "MCMC.hpp"
#include "MarkovChain/markovState.tpp"
#include "NSL.hpp"
#include "complex.hpp"


int main(){

    // Define the parameters of your system (you can also read these in...)
    typedef double Type;

    // Typically you want to read these in or provide as an argument to such 
    // a code but for this example we just specify them here
    // System Parameters
    //    Inverse temperature 
    Type beta = 1;
    //    On-Site Coupling
    Type U    = 1;
    //    Number of time slices
    NSL::size_t Nt = 4;
    //    Number of ions (spatial sites)
    NSL::size_t Nx =  2;

    // Leapfrog Parameters
    //      Trajectory Length
    NSL::RealTypeOf<Type> trajectoryLength = 1.; // We ensure that this is a real number in case Type is complex
    //      Number of Molecular Dynamics steps
    NSL::size_t numberMDsteps = 10;
    
    // Markov Change Parameters 
    //     Number of Burn In configurations to thermalize the chain
    NSL::size_t NburnIn = 100;
    //     Number of configurations to be computed on which we will measure
    NSL::size_t Nconf = 10000;
    //     Number of configurations not used for measurements in between each stored configuration
    NSL::size_t saveFreq = 10;
    // The total number of configurations is given by the product:
    // Nconf_total = Nconf * saveFreq

    // Define the lattice geometry of interest
    // ToDo: Required for more sophisticated actions
    NSL::Lattice::Ring<Type> lattice(Nx); 

    // Put the action parameters into the appropriate container
    NSL::Action::HubbardGaugeAction<Type>::Parameters params(
        /*beta=*/  beta,
        /*Nt = */  Nt,    
        /*U =  */  U
    );

    // Initialize the action
    NSL::Action::Action S{
        NSL::Action::HubbardGaugeAction<Type>(params)
    };

    // Initialize a configuration as starting point for the MC change
    // For CPU code put here
    NSL::Configuration<Type> config{
        {"phi", NSL::Tensor<Type>(Nt,Nx)}
    };
    config["phi"].rand();

    // Initialize the integrator
    NSL::Integrator::Leapfrog leapfrog( 
        /*action*/S,  
        /*trajectoryLength*/trajectoryLength,
        /**numberSteps*/numberMDsteps
    );

    // Initialize the HMC
    NSL::MCMC::HMC hmc(leapfrog, S);

    // Burn In
    // We can pass just a config to the generate function a MarkovState is 
    // generated automatically. If we want more control you cane also provide
    // a MarkovState.
    // The boolean provided as template argument `returnChain` switches 
    // whether the entire chain is stored in a std::vector or of only 
    // a single state is stored at the time. The second option is more 
    // efficient and might be used for e.g. burn in.
    auto burnInStartTime = std::chrono::steady_clock::now();
    NSL::MCMC::MarkovState<Type> start_state = hmc.generate<false>(config, NburnIn);
    auto burnInEndTime = std::chrono::steady_clock::now();

    // Generate Markov Chain
    // Here we should store the entire chain (i.e. every saveFreq element)
    // This generates Nconf*saveFreq configurations, though the std::vector
    // which is returned is of length Nconf.
    // 
    // Note: This also has a overload for providing a configuration only.
    auto productionStartTime = std::chrono::steady_clock::now();
    std::vector<NSL::MCMC::MarkovState<Type>> markovChain = hmc.generate<true>(start_state, Nconf, saveFreq);
    auto productionEndTime = std::chrono::steady_clock::now();

    // Print some final statistics
    std::cout << "Acceptance Rate: " << NSL::MCMC::getAcceptenceRate(markovChain) * 100 << "%" << std::endl;
    std::cout << "Burn In took   : " << std::chrono::duration_cast<std::chrono::nanoseconds>(burnInEndTime - burnInStartTime).count() * 1e-9 << " s" << std::endl;
    std::cout << "Production took: " << std::chrono::duration_cast<std::chrono::nanoseconds>(productionEndTime - productionStartTime).count() * 1e-9 << " s" << std::endl;

    return EXIT_SUCCESS;
}
