#include <chrono>
#include "NSL.hpp"

int main(int argc, char* argv[]){

    NSL::Logger::init_logger(argc, argv);

    // Define the device to run on NSL::GPU(ID=0) or NSL::CPU(ID=0)
    auto total_time =  NSL::Logger::start_profile("Program Start");
    auto init_time =  NSL::Logger::start_profile("Program Initialization");
    auto device = NSL::GPU();

    // Define the parameters of your system (you can also read these in...)
    typedef NSL::complex<double> Type;

    // Typically you want to read these in or provide as an argument to such 
    // a code but for this example we just specify them here
    // System Parameters
    //    Inverse temperature 
    Type beta = 1.0;
    //    On-Site Coupling
    Type U    = 2.0;
    //    Number of time slices
    NSL::size_t Nt = 22;
    //    Number of ions (spatial sites)
    NSL::size_t L1 = 6;
    NSL::size_t L2 = 6;
    std::vector<int> L = {L1,L2};

    
    // Leapfrog Parameters
    //      Trajectory Length
    NSL::RealTypeOf<Type> trajectoryLength = 1.; // We ensure that this is a real number in case Type is complex
    //      Number of Molecular Dynamics steps
    NSL::size_t numberMDsteps = 30;
    
    // Markov Change Parameters 
    //     Number of Burn In configurations to thermalize the chain
    NSL::size_t NburnIn = 1000;
    //     Number of configurations to be computed on which we will measure
    NSL::size_t Nconf = 1000;
    //     Number of configurations not used for measurements in between each stored configuration
    NSL::size_t saveFreq = 1;
    // The total number of configurations is given by the product:
    // Nconf_total = Nconf * saveFreq

    // Define the lattice geometry of interest
    // ToDo: Required for more sophisticated actions
    NSL::Lattice::Honeycomb<Type> lattice(L);

    // Put the lattice on the device. (copy to GPU)
    lattice.to(device);

    // get number of ions
    NSL::size_t Nx = lattice.sites();

    std::string H5NAME(
        fmt::format("./Honeycomb_Nt{}_L1{}_L2{}_U{}_B{}.h5", Nt, L1, L2, NSL::to_string(U), NSL::to_string(beta))
    );  // name of h5 file to store configurations, measurements, etc. . .
    NSL::H5IO h5(H5NAME, NSL::File::Truncate);

    NSL::Logger::info("Setting up a Hubbard action with beta={}, Nt={}, U={}, on a {}.", NSL::real(beta), Nt, NSL::real(U), lattice.name());

    // Put the action parameters into the appropriate container
    NSL::Action::HubbardGaugeAction<Type>::Parameters params(
        /*beta=*/  beta,
        /*Nt = */  Nt,    
        /*U =  */  U
    );
    NSL::Action::HubbardFermionAction<Type,decltype(lattice),
        NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>>::Parameters paramsHFM(
        /*beta=*/  beta,
        /*Nt = */  Nt,    
        /*lattice=*/lattice
    );

    NSL::Action::HubbardGaugeAction<Type> S_gauge(params);
    NSL::Action::HubbardFermionAction<Type,decltype(lattice),NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>> S_fermion(paramsHFM);

    // Initialize the action
    NSL::Action::Action S = S_gauge + S_fermion;

    // Initialize a configuration as starting point for the MC change
    // For CPU code put here
    NSL::Configuration<Type> config{
        {"phi", NSL::Tensor<Type>(device,Nt,Nx)}
    };
    config["phi"].randn();
    config["phi"].imag() = 0;
    config["phi"] *= params.Utilde;

    NSL::Logger::info("Setting up a leapfrog integrator with trajectory length {} and {} MD steps.", trajectoryLength, numberMDsteps);

    // Initialize the integrator
    NSL::Integrator::Leapfrog leapfrog( 
        /*action*/S,  
        /*trajectoryLength*/trajectoryLength,
        /**numberSteps*/numberMDsteps
    );

    // Initialize the HMC
    NSL::MCMC::HMC hmc(leapfrog, S, h5);
    NSL::Logger::stop_profile(init_time);

    // Burn In
    // We can pass just a config to the generate function a MarkovState is 
    // generated automatically. If we want more control you cane also provide
    // a MarkovState.
    // The Template argument Chain{AllStates,LastState} is a memory optimization
    // where the LastState will return only the last generated state and does 
    // not keep the rest in memory while the AllStates (see production for use)
    // will store all states according to the saveFrequency.

    auto therm_time =  NSL::Logger::start_profile("Thermalization");
    NSL::Logger::info("Thermalizing {} steps...", NburnIn);
    NSL::MCMC::MarkovState<Type> start_state = hmc.generate<NSL::MCMC::Chain::LastState>(config, NburnIn);
    NSL::Logger::stop_profile(therm_time);

    // Generate Markov Chain
    // Here we should store the entire chain (i.e. every saveFreq element)
    // This generates Nconf*saveFreq configurations, though the std::vector
    // which is returned is of length Nconf.
    // 
    // Note: This also has a overload for providing a configuration only.
    auto gen_time =  NSL::Logger::start_profile("Generation");
    NSL::Logger::info("Generating {} steps, saving every {}...", Nconf, saveFreq);
    std::vector<NSL::MCMC::MarkovState<Type>> markovChain = hmc.generate<NSL::MCMC::Chain::AllStates>(start_state, Nconf, saveFreq);
    NSL::Logger::stop_profile(gen_time);

    // Print some final statistics
    NSL::Logger::info("Acceptance Rate: {}%", NSL::MCMC::getAcceptanceRate(markovChain) * 100);
    NSL::Logger::stop_profile(total_time);

    return EXIT_SUCCESS;
}
