#include "NSL.hpp"

int main(int argc, char** argv){
    typedef NSL::complex<double> Type;

    NSL::Parameter params = NSL::init(argc,argv,"NSL Hubbard Correlator");
    
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
    // Number of Sources for the cg
    if (yml["measurements"]["Number Time Sources"]){
        params.addParameter<NSL::size_t>(
            "Number Time Sources", yml["measurements"]["Number Time Sources"].as<NSL::size_t>()
        );
    } else {
        // DEFAULT: Number Time Sources = Nt
        params.addParameter<NSL::size_t>(
	    "Number Time Sources", params["Nt"]
        );
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
    NSL::H5IO h5(params["h5file"].to<std::string>(), params["overwrite"].to<bool>());

    // define the basenode for the h5file, everything is stored in 
    // params["h5Filename"]/BASENODE/
    std::string BASENODE(fmt::format("{}",params["name"].repr()));

    // initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);
    NSL::size_t dim = lattice.sites();

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);

    // add the lattice to the parameter
    params.addParameter<decltype(lattice)>("lattice", lattice);

    // initialize 2 point correlation function <p^+_x p_y> 
    NSL::Measure::Hubbard::TwoPointCorrelator<
        Type,
        decltype(lattice),
        NSL::FermionMatrix::HubbardExp<
            Type,decltype(lattice)
        >
    > C2pt_sp(params, h5, NSL::Hubbard::Particle);

    // Perform the measurement.
    // 1. Calculate <p^+_x p_y> = \sum_{ts} < M^{-1}_{t-t_s,x;0;y } >
    //
    // configurations from the data file specified under params["file"].
    // Then 
    C2pt_sp.measure();

    // initialize 2 point correlation function <h^+_x h_y> 
    NSL::Measure::Hubbard::TwoPointCorrelator<
        Type,
        decltype(lattice),
        NSL::FermionMatrix::HubbardExp<
            Type,decltype(lattice)
        >
    > C2pt_sh(params, h5, NSL::Hubbard::Hole);

    // Perform the measurement.
    // 1. Calculate <h^+_x h_y> = \sum_{ts} < M^{-1}_{t-t_s,x;0;y } >
    //
    // configurations from the data file specified under params["file"].
    // Then 
    C2pt_sh.measure();
}
