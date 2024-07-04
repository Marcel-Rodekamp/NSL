#include "NSL.hpp"

int main(int argc, char* argv[]){
    typedef NSL::complex<double> Type;

    NSL::Parameter params = NSL::init(argc, argv, "Schwinger Model HMC");

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
    NSL::H5IO h5(params["h5file"].to<std::string>(),params["overwrite"].to<bool>());

    // define the basenode for the h5file, everything is stored in 
    // params["h5Filename"]/BASENODE/
    std::string BASENODE(fmt::format("{}",params["name"]));


    NSL::Measure::U1::chiralCondensate<Type> CC( params, h5, BASENODE);
    CC.measure();



   
    return EXIT_SUCCESS;
}
