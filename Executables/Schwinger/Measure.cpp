#include "NSL.hpp"

int main(int argc, char* argv[]){
    typedef NSL::complex<double> Type;

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
    NSL::H5IO h5(params["h5file"].to<std::string>(),params["overwrite"].to<bool>());

    // define the basenode for the h5file, everything is stored in 
    // params["h5Filename"]/BASENODE/
    std::string BASENODE(fmt::format("{}",params["name"].repr()));


    for( NSL::size_t nx = 1; nx < params["Nx"].to<NSL::size_t>() / 2+1; ++nx) { 
        for( NSL::size_t nt = 1; nt < params["Nt"].to<NSL::size_t>() / 2+1; ++nt) { 
            NSL::Measure::U1::PlanarWilsonLoop<Type> L(
                params, h5, 
                /*N_t*/nt, 
                /*N_x*/nx,
                BASENODE
            );
        
            L.measure();
        }
    }

   
    return EXIT_SUCCESS;
}
