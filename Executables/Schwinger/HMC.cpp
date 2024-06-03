#include "NSL.hpp"

template<NSL::Concept::isNumber Type>
void writeMeta(NSL::Parameter & params, NSL::H5IO & h5, std::string BASENODE);

int main(int argc, char* argv[]){
    typedef NSL::complex<double> Type;

    NSL::complex<double> I{0,1};

    NSL::Parameter params = NSL::init(argc, argv, "Schwinger Model HMC");

    YAML::Node yml = YAML::LoadFile(params["file"]);

    params.addParameter<std::string>(
        "name", yml["system"]["name"].as<std::string>()
    );
    // The inverse temperature 
    params.addParameter<Type>(
        "beta", yml["system"]["beta"].as<double>()
    );
    params.addParameter<Type>(
        "bare mass", yml["system"]["bare mass"].as<double>()
    );
    // The number of time slices
    params.addParameter<NSL::size_t>(
        "Nt", yml["system"]["Nt"].as<NSL::size_t>()
    );
    // The number of ions
    params.addParameter<NSL::size_t>(
        "Nx", yml["system"]["Nx"].as<NSL::size_t>()
    );
    // The HMC save frequency
    params.addParameter<NSL::size_t>(
        "save frequency", yml["HMC"]["save frequency"].as<NSL::size_t>()
    );
    // The thermalization length
    params.addParameter<NSL::size_t>(
        "Ntherm", yml["HMC"]["Ntherm"].as<NSL::size_t>()
    );
    // The number of configurations
    params.addParameter<NSL::size_t>(
        "Nconf", yml["HMC"]["Nconf"].as<NSL::size_t>()
    );
    // The trajectory length
    params.addParameter<double>(
        "trajectory length", yml["Leapfrog"]["trajectory length"].as<double>()
    );
    // The number of molecular dynamic steps
    params.addParameter<NSL::size_t>(
        "Nmd", yml["Leapfrog"]["Nmd"].as<NSL::size_t>()
    );
    // The h5 file name to store the simulation results
    params.addParameter<std::string>(
        "h5file", yml["fileIO"]["h5file"].as<std::string>()
    );

    params.addParameter<NSL::size_t>("dim",2);

    NSL::Lattice::Square<Type> lattice({
        params["Nt"].to<NSL::size_t>(),
        params["Nx"].to<NSL::size_t>()
    });
    lattice.to(params["device"].to<NSL::Device>());

    params.addParameter<NSL::Lattice::Square<Type>>("lattice",lattice);

    for(auto [key, value]: params){
        // skip these keys as they are logged in init already
        if (key == "device" || key == "file") {continue;}
        NSL::Logger::info( "{}: {}", key, value->repr() );
    }

    // create an H5 object to store data
    //NSL::H5IO h5(params["h5file"].to<std::string>(),params["overwrite"].to<bool>());
    NSL::H5IO h5(params["h5file"].to<std::string>(),NSL::File::Truncate);

    // define the basenode for the h5file, everything is stored in 
    // params["h5Filename"]/BASENODE/
    std::string BASENODE(fmt::format("{}",params["name"].repr()));

    // write the meta data to the h5file
    writeMeta<Type>(params, h5, BASENODE);

    NSL::Logger::info("Setting up: Schwinger Model Action with Wilson fermion and gauge") ;

    NSL::Action::Action S = NSL::Action::PseudoFermionAction<
                                Type,
                                decltype(lattice),
                                NSL::FermionMatrix::U1::Wilson<Type>
                            >(params,"U")
                          + NSL::Action::U1::WilsonGaugeAction<Type>(params)
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
    S.computePseudoFermion(config);

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
    h5.write(params["save frequency"].to<NSL::size_t>(), fmt::format("{}/Meta/params/{}",basenode,"save frequency"));
    h5.write(params["Ntherm"].to<NSL::size_t>(), fmt::format("{}/Meta/params/{}",basenode,"Ntherm"));
    h5.write(params["Nconf"].to<NSL::size_t>(), fmt::format("{}/Meta/params/{}",basenode,"Nconf"));
    h5.write(params["trajectory length"].to<double>(), fmt::format("{}/Meta/params/{}",basenode,"trajectory length"));
    h5.write(params["Nmd"].to<NSL::size_t>(), fmt::format("{}/Meta/params/{}",basenode,"Nmd"));
}
