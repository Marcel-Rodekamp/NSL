#include "NSL.hpp"

template<NSL::Concept::isNumber Type>
void writeMeta(NSL::Parameter & params, NSL::H5IO & h5, std::string BASENODE);

int main(int argc, char* argv[]){
    typedef NSL::complex<double> Type;
    NSL::complex<double> I{0,1};

    // Initialize NSL
    NSL::Parameter params = NSL::init(argc, argv, "Schwinger Model HMC");
    // an example parameter file is can be found in example_param.yml
    
    // Now all parameters are stored in yml, we want to translate them 
    // into the parameter object
    // We can read in the parameter file and put the read data into the 
    // params object, notice this uses the example_param.yml file
    // For personal files, this code needs to be adjusted accordingly
    YAML::Node yml = YAML::LoadFile(params["file"]);

    // convert the data from example_param.yml and put it into the params
    // The name of the physical system
    params["name"]              = yml["system"]["name"].as<std::string>();
    // The inverse temperature 
    params["beta"]              = yml["system"]["beta"].as<double>();
    // bare mass 
    params["bare mass"]         = yml["system"]["bare mass"].as<double>();
    // The number of time slices
    params["Nt"] = yml["system"]["Nt"].as<NSL::size_t>();
    // The number of ions
    params["Nx"] = yml["system"]["Nx"].as<NSL::size_t>();
    params["Nf"] = yml["system"]["Nf"].as<NSL::size_t>();
    // The HMC save frequency
    params["save frequency"] = yml["HMC"]["save frequency"].as<NSL::size_t>();
    // The thermalization length
    params["Ntherm"] = yml["HMC"]["Ntherm"].as<NSL::size_t>();
    // The number of configurations
    params["Nconf"] = yml["HMC"]["Nconf"].as<NSL::size_t>();
    // The trajectory length
    params["trajectory length"] = yml["Leapfrog"]["trajectory length"].as<double>();
    // The number of molecular dynamic steps
    params["Nmd"] = yml["Leapfrog"]["Nmd"].as<NSL::size_t>();
    // The h5 file name to store the simulation results
    params["h5file"] = yml["fileIO"]["h5file"].as<std::string>();
    // dimension of the system 
    params["dim"] = 2;

    NSL::Lattice::Square<Type> lattice({
        params["Nt"].to<NSL::size_t>(),
        params["Nx"].to<NSL::size_t>()
    });
    lattice.to(params["device"].to<NSL::Device>());

    for(auto [key, value]: params){
        // skip these keys as they are logged in init already
        if (key == "device" || key == "file") {continue;}
        NSL::Logger::info( "{}: {}", key, value );
    }

    // create an H5 object to store data
    //NSL::H5IO h5(params["h5file"].to<std::string>(),params["overwrite"].to<bool>());
    NSL::H5IO h5(params["h5file"].to<std::string>(),NSL::File::Truncate);

    // define the basenode for the h5file, everything is stored in 
    // params["h5Filename"]/BASENODE/
    std::string BASENODE(fmt::format("{}",params["name"]));

    // write the meta data to the h5file
    writeMeta<Type>(params, h5, BASENODE);

    NSL::Logger::info("Setting up: Schwinger Model Action with Wilson fermion and gauge") ;

       NSL::Action::Action S =
                             NSL::Action::WilsonFermionAction<
                             Type,decltype(lattice), 
                             NSL::FermionMatrix::U1::Wilson<Type>
                             >(lattice,params,"U")
                        + 
                        NSL::Action::U1::WilsonGaugeAction<Type>(params)
                           ;

    // Initialize a configuration as starting point for the MC change
    NSL::Configuration<Type> config;
    config["U"] = NSL::LinAlg::exp( I *NSL::randn<NSL::RealTypeOf<Type>>(
        params["device"].to<NSL::Device>(),
        params["Nt"].to<NSL::size_t>(),
        params["Nx"].to<NSL::size_t>(),
        params["dim"].to<NSL::size_t>()
    ) );

    // compute pseudo fermions (if they don't exist this call does nothing)
    //S.computePseudoFermion(config);

    NSL::Logger::info("Setting up: Leapfrog integrator");

    // Initialize the integrator defining the equation of motion via the 
    // real part of the force (real part of the action)
    NSL::Integrator::U1::Leapfrog leapfrog(
        /*action*/S,  
        /*trajectoryLength*/NSL::RealTypeOf<Type>(params["trajectory length"]),
        /**numberSteps*/NSL::size_t(params["Nmd"])
    );

    NSL::Logger::info("Setting up: HMC");
    NSL::MCMC::HMC hmc(leapfrog, S, h5);
    
    NSL::Logger::info("Thermalizing {} steps...", params["Ntherm"].to<NSL::size_t>());
    NSL::MCMC::MarkovState<Type> start_state = hmc.generate<
        NSL::MCMC::Chain::LastState
    >(config, params["Ntherm"].to<NSL::size_t>());
 
    NSL::Logger::info("Generating {} steps, saving every {}...", params["Nconf"].to<NSL::size_t>(), params["save frequency"].to<NSL::size_t>());
    std::vector<NSL::MCMC::MarkovState<Type>> markovChain = hmc.generate<
        NSL::MCMC::Chain::AllStates
    >(start_state, params["Nconf"], params["save frequency"], BASENODE+"/markovChain");

    NSL::Logger::info("Acceptance Rate: {}%", NSL::MCMC::getAcceptanceRate(markovChain) * 100);

    h5.write(NSL::MCMC::getAcceptanceRate(markovChain), fmt::format("{}/Meta/acceptenceRate",BASENODE));
    
    return EXIT_SUCCESS;
}

template<NSL::Concept::isNumber Type>
void writeMeta(NSL::Parameter & params, NSL::H5IO & h5, std::string basenode){
    
    h5.write(params["beta"].to<Type>(), fmt::format("{}/Meta/params/{}",basenode,"beta"));
    h5.write(params["bare mass"].to<Type>(), fmt::format("{}/Meta/params/{}",basenode,"bare mass"));
    h5.write(params["Nt"].to<NSL::size_t>(), fmt::format("{}/Meta/params/{}",basenode,"Nt"));
    h5.write(params["Nx"].to<NSL::size_t>(), fmt::format("{}/Meta/params/{}",basenode,"Nx"));
    h5.write(params["Nf"].to<NSL::size_t>(), fmt::format("{}/Meta/params/{}",basenode,"Nf"));
    h5.write(params["save frequency"].to<NSL::size_t>(), fmt::format("{}/Meta/params/{}",basenode,"save frequency"));
    h5.write(params["Ntherm"].to<NSL::size_t>(), fmt::format("{}/Meta/params/{}",basenode,"Ntherm"));
    h5.write(params["Nconf"].to<NSL::size_t>(), fmt::format("{}/Meta/params/{}",basenode,"Nconf"));
    h5.write(params["trajectory length"].to<double>(), fmt::format("{}/Meta/params/{}",basenode,"trajectory length"));
    h5.write(params["Nmd"].to<NSL::size_t>(), fmt::format("{}/Meta/params/{}",basenode,"Nmd"));
}
