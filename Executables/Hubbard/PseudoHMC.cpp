#include "NSL.hpp"

template<NSL::Concept::isNumber Type, typename LatticeType>
void writeMeta(LatticeType lat, NSL::Parameter & params, NSL::H5IO & h5, std::string BASENODE);

int main(int argc, char* argv[]){
    typedef NSL::complex<double> Type;
    NSL::complex<double> I{0,1};

    // Initialize NSL
    NSL::Parameter params = NSL::init(argc, argv, "Hubbard Model HMC with Pseudofermions");
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
    // The number of time slices
    params["Nt"]                = yml["system"]["Nt"].as<NSL::size_t>();
    // The number of ions
    params["Nx"]                = yml["system"]["nions"].as<NSL::size_t>();
    // The on-site interaction
    params["U"]                 = yml["system"]["U"].as<double>();
    // The HMC save frequency
    params["save frequency"]    = yml["HMC"]["save frequency"].as<NSL::size_t>();
    // The thermalization length
    params["Ntherm"]            = yml["HMC"]["Ntherm"].as<NSL::size_t>();
    // The number of configurations
    params["Nconf"]             = yml["HMC"]["Nconf"].as<NSL::size_t>();
    // The trajectory length
    params["trajectory length"] = yml["Leapfrog"]["trajectory length"].as<double>();
    // The number of molecular dynamic steps
    params["Nmd"]               = yml["Leapfrog"]["Nmd"].as<NSL::size_t>();
    // The h5 file name to store the simulation results
    params["h5file"]            = yml["fileIO"]["h5file"].as<std::string>();
    // The offset: tangent plane/NLO plane
    if (yml["system"]["offset"]){
        params["offset"]        = yml["system"]["offset"].as<double>();
    } else {
        // DEFAULT: offset = 0
        params["offset"]        = 0.0;
    }
    // Chemical Potential
    if (yml["system"]["mu"]){
        params["mu"]            = yml["system"]["mu"].as<double>();
        if (params["mu"].to<double>() != 0.0){
            NSL::Logger::error("Chemical potential is not implemented yet.");
            return EXIT_FAILURE;
        }
    } else {
        // DEFAULT: mu = 0
        params["mu"]            = 0.0;
    }

    for(auto [key, value]: params){
        // skip these keys as they are logged in init already
        if (key == "device" || key == "file") {continue;}
        NSL::Logger::info( "{}: {}", key, value );
    }

    // create an H5 object to store data
    NSL::H5IO h5(
        params["h5file"].to<std::string>(), 
        params["overwrite"].to<bool>() ? NSL::File::Truncate : NSL::File::ReadWrite | NSL::File::OpenOrCreate
    );

    // define the basenode for the h5file, everything is stored in 
    // params["h5Filename"]/BASENODE/
    std::string BASENODE(fmt::format("{}",params["name"]));

    // initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);
    if (lattice.sites() != params["Nx"].template to<NSL::size_t>()){
        throw std::runtime_error("The number of ions in the parameter file does not match the number of ions in the lattice.");
    }

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);

    // write the meta data to the h5file
    writeMeta<Type,decltype(lattice)>(lattice, params, h5, BASENODE);

    NSL::Logger::info("Setting up a Hubbard action with beta={}, Nt={}, U={}, on a {}.", 
        params["beta"],
        params["Nt"],
        params["U"],
        params["name"]
    );

    NSL::Action::Action S = NSL::Action::HubbardGaugeAction<Type>(params)
                            + NSL::Action::PseudoFermionAction<
                            Type,decltype(lattice), NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>
                            >(lattice, params)
    ;

    // Initialize a configuration as starting point for the MC change
    NSL::Configuration<Type> config{
        {"phi", 
            NSL::Tensor<Type>(
                NSL::Device(params["device"]),
                NSL::size_t(params["Nt"]),
                NSL::size_t(params["Nx"])
            )
        }
    };
    config["phi"].randn();
    config["phi"] *= NSL::Hubbard::tilde<Type>(params, "U");
    config["phi"].imag() = NSL::RealTypeOf<Type>(params["offset"]);

    // compute pseudo fermions (if they don't exist this call does nothing)
    S.computePseudoFermion(config);

    NSL::Logger::info("Setting up: Leapfrog integrator");

    // Initialize the integrator defining the equation of motion via the 
    // real part of the force (real part of the action)
    NSL::Integrator::LeapfrogRealForce leapfrog( 
        /*action*/S,  
        /*trajectoryLength*/NSL::RealTypeOf<Type>(params["trajectory length"]),
        /**numberSteps*/params["Nmd"]
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

template<NSL::Concept::isNumber Type, typename LatticeType>
void writeMeta(LatticeType lat, NSL::Parameter & params, NSL::H5IO & h5, std::string BASENODE){
    // write out the physical and run parameters for this system
    HighFive::File h5file = h5.getFile();

    if(h5.exist(BASENODE+"/Meta")){
        NSL::Logger::info("Meta data already exists, skipping.");
        return;
    }

    // lattice name
    HighFive::DataSet dataset = h5file.createDataSet<std::string>(
        BASENODE+"/Meta/lattice",
        HighFive::DataSpace::From(lat.name())
    );
    dataset.write(lat.name());

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
        HighFive::DataSpace::From(lat.sites())
    );
    dataset.write(lat.sites());

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
