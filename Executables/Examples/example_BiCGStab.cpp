#include "NSL.hpp"
#include "logger.hpp"

int main(int argc, char ** argv){
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
    
    // initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);
    NSL::size_t dim = lattice.sites();

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);

    params.addParameter<decltype(lattice)>("lattice", lattice);


    // get number of ions
    NSL::size_t Nx = lattice.sites();

    NSL::Logger::info("Setting up a Hubbard action with beta={}, Nt={}, U={}, on a {}.",
        NSL::real( Type(params["beta"]) ),
        params["Nt"].to<NSL::size_t>(),
        NSL::real( Type(params["U"])),
        std::string( params["name"] )
    );

    NSL::Tensor<NSL::complex<double>> phi(params["device"].to<NSL::Device>(), params["Nt"].to<NSL::size_t>(),params["Nx"].to<NSL::size_t>());
    NSL::FermionMatrix::HubbardExp M(lattice,params["Nt"].to<NSL::size_t>(),params["beta"].to<Type>());
    M.populate(phi,NSL::Hubbard::Species::Particle);

    NSL::Logger::info("Running BiCGStab to compute MM†@x = b.");

    NSL::LinAlg::BiCGStab<NSL::complex<double>> bicgstab(M,NSL::FermionMatrix::MMdagger, 1e-10);

    NSL::Tensor<NSL::complex<double>> b = NSL::zeros_like(phi);
    b(NSL::Slice(0,1), NSL::Slice(0,1)) = 1;

    auto bicgstab_time =  NSL::Logger::start_profile("BiCGStab");
    for(int i = 0; i < 100; i++)
        NSL::Tensor<NSL::complex<double>> res = bicgstab(b);
    NSL::Logger::stop_profile(bicgstab_time);
    NSL::Logger::info("BiCGStab took {}s ",std::get<0>(bicgstab_time));

    NSL::Logger::info("Done");
}

