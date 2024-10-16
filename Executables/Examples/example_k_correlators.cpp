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
    } else {
        // DEFAULT: mu = 0
        params["mu"]            = 0.0;
    }
    // Number of Sources for the cg
    if (yml["measurements"]["Number Time Sources"]){
        params["Number Time Sources"] = yml["measurements"]["Number Time Sources"].as<NSL::size_t>();
    } else {
        // DEFAULT: Number Time Sources = Nt
        params["Number Time Sources"] = params["Nt"];
    }

    // Now we want to log the found parameters
    // - key is a std::string name,beta,...
    // - value is a ParameterEntry * which is a wrapper around the actual 
    //   value of interest, we can use ParameterEntry::repr() to get a string
    //   representation of the stored value
    for(auto [key, value]: params){
        // skip these keys as they are logged in init already
        if (key == "device" || key == "file") {continue;}
        NSL::Logger::info( "{}: {}", key, value );
    }

    std::vector<std::vector<std::vector<std::vector<double>>>> kblocks2d;
    std::vector<std::vector<double>> momenta;
   

    // Load momentum blocks if they exist
    if (yml["measurements"]["momenta"]){
      momenta = yml["measurements"]["momenta"].as<std::vector<std::vector<double>>>();
      NSL::Tensor<double> mblocks(momenta.size(),momenta[0].size());
      for (int i=0;i<momenta.size(); i++){
	for (int j=0;j<momenta[0].size();j++){
	  mblocks(i,j) = momenta[i][j];
	}
      }
      params["momenta"]=mblocks;
    }
    if (yml["measurements"]["wallSources"]){
      kblocks2d = yml["measurements"]["wallSources"].as<std::vector<std::vector<std::vector<std::vector<double>>>>>();
      NSL::Tensor<NSL::complex<double>> kblocks(kblocks2d.size(),kblocks2d[0].size(),kblocks2d[0][0].size());
      NSL::Logger::info( "Measuring {} momentum block(s), each with {} band(s) of length {}",kblocks2d.size(),kblocks2d[0].size(),kblocks2d[0][0].size());

      // now populate the momentum blocks with proper complex variables
      for (int i=0; i<kblocks2d.size(); i++){
	for (int j=0; j<kblocks2d[0].size(); j++) {
	  for (int k=0; k<kblocks2d[0][0].size(); k++) {
	    kblocks(i,j,k) = NSL::complex<double> (kblocks2d[i][j][k][0], kblocks2d[i][j][k][1]);
	  }
	}
      }
      params["wallSources"]=kblocks.to(params["device"]);
    } else {
      // DEFAULT: raise an exception
      // currently don't know how to do this, will do later
    }

    
    //    exit(0);
    
    // create an H5 object to store data
    NSL::H5IO h5(
        params["h5file"].to<std::string>(), 
        params["overwrite"].to<bool>() ? NSL::File::Truncate : NSL::File::ReadWrite | NSL::File::OpenOrCreate
    );

    // define the basenode for the h5file, everything is stored in 
    // params["h5Filename"]/BASENODE/
    std::string BASENODE(fmt::format("{}",std::string(params["name"])));
    // initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);

    // initialize 2 point correlation function <p^+_x p_y> 
    NSL::Measure::Hubbard::TwoPointCorrelator<
        Type,
        decltype(lattice),
        NSL::FermionMatrix::HubbardExp<
            Type,decltype(lattice)
        >
    > C2pt_sp(lattice, params, h5, NSL::Hubbard::Particle);

    // Perform the measurement.
    // 1. Calculate <p^+_x p_y> = \sum_{ts} < M^{-1}_{t-t_s,x;0;y } >
    //
    // configurations from the data file specified under params["file"].
    // Then 
    C2pt_sp.measureK();

    // initialize 2 point correlation function <h^+_x h_y> 
    NSL::Measure::Hubbard::TwoPointCorrelator<
      Type,
      decltype(lattice),
      NSL::FermionMatrix::HubbardExp<
      Type,decltype(lattice)
      >
    > C2pt_sh(lattice, params, h5, NSL::Hubbard::Hole);

    // Perform the measurement.
    // 1. Calculate <h^+_x h_y> = \sum_{ts} < M^{-1}_{t-t_s,x;0;y } >
    //
    // configurations from the data file specified under params["file"].
    // Then 
    C2pt_sh.measureK();
}
