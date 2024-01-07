#include <chrono>
#include "NSL.hpp"
#include "highfive/H5File.hpp"
#include <yaml-cpp/yaml.h>

#include "transformers.tpp"

template<NSL::Concept::isNumber Type, typename LatticeType>
void writeMeta(NSL::Parameter & params, NSL::H5IO & h5, std::string BASENODE);

int main(int argc, char* argv[]){
    typedef NSL::complex<double> Type;

    // Initialize NSL
    NSL::Parameter params = NSL::init(argc, argv, "Example MCMC");
    // an example parameter file is can be found in example_param.yml
    
    auto init_time = NSL::Logger::start_profile("Initialization");
    
    // Now all parameters are stored in yml, we want to translate them 
    // into the parameter object
    // We can read in the parameter file and put the read data into the 
    // params object, notice this uses the example_param.yml file
    // For personal files, this code needs to be adjusted accordingly
    YAML::Node yml = YAML::LoadFile(params["file"]);

    // convert the data from example_param.yml and put it into the params
    // The name of the physical system
    params.addParameter<std::string>(
        "name", yml["system"]["name"].as<std::string>()
    );
    // The inverse temperature 
    params.addParameter<Type>(
        "beta", yml["system"]["beta"].as<double>()
    );
    // The number of time slices
    params.addParameter<NSL::size_t>(
        "Nt", yml["system"]["Nt"].as<NSL::size_t>()
    );
    // The number of ions
    params.addParameter<NSL::size_t>(
        "Nx", yml["system"]["nions"].as<NSL::size_t>()
    );
    // The on-site interaction
    params.addParameter<Type>(
        "U", yml["system"]["U"].as<double>()
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

    params.addParameter<double>(
        "offset", yml["system"]["offset"].as<double>()
    );
    
    // ==================================================================================
    // These are optional parameters, if yml does not contain the node the default value
    // will be used. 
    // ==================================================================================
        
    // Chemical Potential
    if (yml["system"]["mu"]){
        params.addParameter<Type>(
            "mu", yml["system"]["mu"].as<double>()
        );
    } else {
        // DEFAULT: mu = 0
        params.addParameter<Type>("mu");
    }

    // Now we want to log the found parameters
    // - key is a std::string name,beta,...
    // - value is a ParameterEntry * which is a wrapper around the actual 
    //   value of interest, we can use ParameterEntry::repr() to get a string
    //   representation of the stored value
    for(auto [key, value]: params){
        // skip these keys as they are logged in init already
        if (key == "device" || key == "file") {continue;}
        NSL::Logger::info( "{}: {}", key, value->repr() );
    }

    // create an H5 object to store data
    NSL::H5IO h5(
        params["h5file"].to<std::string>(), 
        params["overwrite"].to<bool>() ? NSL::File::Truncate : NSL::File::ReadWrite | NSL::File::OpenOrCreate
    );

    // define the basenode for the h5file, everything is stored in 
    // params["h5Filename"]/BASENODE/
    std::string BASENODE(fmt::format("{}",params["name"].repr()));

    // initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);
    NSL::size_t dim = lattice.sites();

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);

    params.addParameter<decltype(lattice)>("lattice", lattice);

    // write the meta data to the h5file
    writeMeta<Type,decltype(lattice)>(params, h5, BASENODE);

    // get number of ions
    NSL::size_t Nx = lattice.sites();

    NSL::Logger::info("Setting up a Hubbard action with beta={}, Nt={}, U={}, on a {}.", 
        NSL::real( Type(params["beta"]) ),
        params["Nt"].to<NSL::size_t>(),
        NSL::real( Type(params["U"])),
        std::string( params["name"] )
    );

    // define a hubbard gauge action
    NSL::Action::HubbardGaugeAction<Type> S_gauge(params);

    // define a hubbard fermion action, the discretization (HubbardExp) is
    // hard wired in the meta data if you change this here, also change the
    // writeMeta()
    NSL::Action::HubbardFermionAction<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>
    > S_fermion(params);

    // Initialize the action being the sum of the gauge action & fermion action
    NSL::Action::Action S = S_gauge + S_fermion;

    // Initialize a configuration as starting point for the MC change
    NSL::Configuration<Type> config{
        {"phi", NSL::Tensor<Type>(params["device"].to<NSL::Device>(),params["Nt"].to<NSL::size_t>(),Nx)}
    };

    //! \todo: we really need a proper random interface...
    config["phi"].randn();
    config["phi"] *= NSL::Hubbard::tilde<Type>(params, "U");
    config["phi"].imag() = params["offset"].to<double>();
    
    NSL::Logger::info("Setting up a leapfrog integrator with trajectory length {} and {} MD steps.", params["trajectory length"].repr(), params["Nmd"].repr());

    // Initialize the integrator defining the equation of motion via the 
    // real part of the force (real part of the action)
    NSL::Integrator::LeapfrogRealForce leapfrog( 
        /*action*/S,  
        /*trajectoryLength*/NSL::RealTypeOf<Type>(params["trajectory length"]),
        /**numberSteps*/NSL::size_t(params["Nmd"])
    );

    NSL::TorchTransformer<Type, ConstantShift<Type>> transformer(
        ConstantShift<Type>( 0.0 )
    );

    // Initialize the HMC
    NSL::MCMC::TransformerHMC<
        decltype(leapfrog),decltype(S),decltype(transformer)
    > hmc(leapfrog, S, transformer, h5);
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
    NSL::MCMC::MarkovState<Type> start_state;
    if(not h5.exist(BASENODE+"/markovChain")){
        NSL::Logger::info("Thermalizing {} steps...", params["Ntherm"].to<NSL::size_t>());
        start_state = hmc.generate<NSL::MCMC::Chain::LastState>(config, params["Ntherm"].to<NSL::size_t>());
    } else {
        NSL::Logger::info("Appending to previous data.");
        start_state = hmc.generate<NSL::MCMC::Chain::LastState>(config, 1);
    }
    
    NSL::Logger::stop_profile(therm_time);

    // Generate Markov Chain
    // Here we should store the entire chain (i.e. every saveFreq element)
    // This generates Nconf*saveFreq configurations, though the std::vector
    // which is returned is of length Nconf.
    // 
    // Note: This also has a overload for providing a configuration only.
    auto gen_time =  NSL::Logger::start_profile("Generation");
    NSL::Logger::info("Generating {} steps, saving every {}...", params["Nconf"].to<NSL::size_t>(), params["save frequency"].to<NSL::size_t>());
    std::vector<NSL::MCMC::MarkovState<Type>> markovChain = hmc.generate<NSL::MCMC::Chain::AllStates>(start_state, params["Nconf"], params["save frequency"], BASENODE+"/markovChain");
    NSL::Logger::stop_profile(gen_time);

    // Print some final statistics
    NSL::Logger::info("Acceptance Rate: {}%", NSL::MCMC::getAcceptanceRate(markovChain) * 100);

    return EXIT_SUCCESS;
}

template<NSL::Concept::isNumber Type, typename LatticeType>
void writeMeta(NSL::Parameter & params, NSL::H5IO & h5, std::string BASENODE){
    // write out the physical and run parameters for this system
    HighFive::File h5file = h5.getFile();

    if(h5.exist(BASENODE+"/Meta")){
        NSL::Logger::info("Meta data already exists, skipping.");
        return;
    }

    // lattice name
    HighFive::DataSet dataset = h5file.createDataSet<std::string>(
        BASENODE+"/Meta/lattice",
        HighFive::DataSpace::From(params["lattice"].to<LatticeType>().name())
    );
    dataset.write(params["lattice"].to<LatticeType>().name());

    // U
    dataset = h5file.createDataSet<std::complex<NSL::RealTypeOf<Type>>>(
        BASENODE+"/Meta/params/U", 
        HighFive::DataSpace::From(static_cast <std::complex<NSL::RealTypeOf<Type>>> (
            Type(params["U"])
        ))
    );
    dataset.write(static_cast <std::complex<NSL::RealTypeOf<Type>>> (Type(params["U"])));
        
    // beta
    dataset = h5file.createDataSet<std::complex<NSL::RealTypeOf<Type>>>(
        BASENODE+"/Meta/params/beta",
        HighFive::DataSpace::From(static_cast <std::complex<NSL::RealTypeOf<Type>>> (
            Type(params["beta"])
        ))
    );
    dataset.write(static_cast <std::complex<NSL::RealTypeOf<Type>>> (Type(params["beta"])));

    // Nt
    dataset = h5file.createDataSet<NSL::size_t>(
        BASENODE+"/Meta/params/Nt",
        HighFive::DataSpace::From(params["Nt"].to<NSL::size_t>())
    );
    dataset.write(params["Nt"].to<NSL::size_t>());

    // dim
    dataset = h5file.createDataSet<NSL::size_t>(
        BASENODE+"/Meta/params/spatialDim",
        HighFive::DataSpace::From(LatticeType(params["lattice"]).sites())
    );
    dataset.write(LatticeType(params["lattice"]).sites());

    // Nmd
    dataset = h5file.createDataSet<NSL::size_t>(
        BASENODE+"/Meta/params/nMD",
        HighFive::DataSpace::From(NSL::size_t(params["Nmd"]))
    );
    dataset.write(NSL::size_t(params["Nmd"]));

    // saveFreq
    dataset = h5file.createDataSet<NSL::size_t>(
        BASENODE+"/Meta/params/saveFreq",
        HighFive::DataSpace::From(NSL::size_t(params["save frequency"])));
    dataset.write(NSL::size_t(params["save frequency"]));

    // trajectory length
    dataset = h5file.createDataSet<NSL::RealTypeOf<Type>>(
        BASENODE+"/Meta/params/trajLength",
        HighFive::DataSpace::From(NSL::RealTypeOf<Type>(params["trajectory length"]))
    );
    dataset.write(NSL::RealTypeOf<Type>(params["trajectory length"]));

    // action type
    std::string action = "hubbardExp";
    dataset = h5file.createDataSet<std::string>(BASENODE+"/Meta/action",HighFive::DataSpace::From(action));
    dataset.write(action);

}
