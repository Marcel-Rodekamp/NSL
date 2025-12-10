#include "Action/Implementations/hubbardGaugeAction.tpp"
#include "Action/Implementations/hubbardFermiAction.tpp"
#include "NSL.hpp"
#include <iostream>
#include <vector>
#include <ctime>

int main(int argc, char* argv[]){

    typedef NSL::complex<double> Type;

    // Initialize NSL
    NSL::Parameter params = NSL::init(argc, argv, "Example MCMC");
    // an example parameter file is RadialMCMC_example_param.yml
    
    auto init_time = NSL::Logger::start_profile("Initialization");
    
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
    // The full algorithm's save frequency; i.e. frequency in combined update steps of Nradial radial updates and Nhmc HMC steps 
    params["save frequency"]    = yml["HMC"]["save frequency"].as<NSL::size_t>();
    // The number of Radial Updates per combined step
    if (yml["HMC"]["Nradial"]){
        params["Nradial"]  = yml["HMC"]["Nradial"].as<NSL::size_t>();
    }
    else {
        params["Nradial"] = 0;
    }
    // The number of HMC steps per combined step
    if (yml["HMC"]["Nhmc"]){
        params["Nhmc"]  = yml["HMC"]["Nhmc"].as<NSL::size_t>();
    }
    else {
        params["Nhmc"] = 1;
    }
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

    // Standard deviation of proposal lognormal distribution in radial udpate
    if (yml["HMC"]["radial scale"]){
        params["radial scale"]    = yml["HMC"]["radial scale"].as<double>();
    } else {
        // DEFAULT: radialScale = 1/Volume
        params["radial scale"]    = 1./(params["Nt"].to<double>()*params["Nx"].to<double>()); 
    }

    if (params["radial scale"].to<double>()==0.0){
        params["Nradial"] = 0;
    }

    // Stability method
    if (yml["stability"] ) {
      params["stability"]    = yml["stability"].as<std::string>();
    }

    // initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);
    if (lattice.sites() != params["Nx"].template to<NSL::size_t>()){
        throw std::runtime_error("The number of ions in the parameter file does not match the number of ions in the lattice.");
    }

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);
    
    // define a hubbard gauge action
    NSL::Action::HubbardGaugeAction<Type> Sg(params);

    // define a hubbard fermion action, the discretization (HubbardExp) is
    // hard wired in the meta data if you change this here, also change the
    // writeMeta()
    //
    NSL::Action::HubbardFermionAction<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>
      > Sf(lattice,params);
    
    // set stability method if defined in yml file, otherwise default is "QR"
    if (yml["stability"] ) {
      std::string stabilityMethod = params["stability"];
      Sf.hfm_.stabilityMethod = stabilityMethod;// "QR", "DIRECTINVERSE", "SVD"
    }

    // Initialize the action being the sum of the gauge action & fermion action
    NSL::Action::Action S = Sg + Sf;

    NSL::size_t Nx =  NSL::size_t(params["Nx"]);
    NSL::size_t Nt =  NSL::size_t(params["Nt"]);


    NSL::Configuration<Type> config{
        {"phi",
            NSL::Tensor<Type>(
                NSL::Device(params["device"]),
                NSL::size_t(params["Nt"]),
                NSL::size_t(params["Nx"])
            )
        }
    };

    // set seed if desired
    NSL::setSeed(1234);
    
    //! \todo: we really need a proper random interface...
    std::complex<double> I = std::complex<double> (0.0,1.0);

    config["phi"].randn();
    config["phi"] *= NSL::Hubbard::tilde<Type>(params, "U");
    if (yml["system"]["offset"]){
      config["phi"].imag() = NSL::RealTypeOf<Type>(params["offset"]);
    } else {
      config["phi"].imag() = 0.0;
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

    // create an H5 object to store data
    NSL::H5IO h5(
        params["h5file"].to<std::string>(), 
        params["overwrite"].to<bool>() ? NSL::File::Truncate : NSL::File::ReadWrite | NSL::File::OpenOrCreate
    );
    
    clock_t ti = clock();

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
    //    C2pt_sp.measure();

    NSL::Tensor<Type> phi;
    int cfgID = 0;
    std::string basenode = params["name"];
    h5.read(phi,fmt::format("{}/markovChain/{}/phi",basenode,cfgID));
    config["phi"]=phi;
    Sf.hfm_.populate(config["phi"], NSL::Hubbard::Species::Particle);
    Sf.hfm_.gradLogDetM();
    //        std::cout << std::setprecision(15) << Sf.hfm_.logDetM() << std::endl;
    //    Sf.hfm_.populate(config["phi"], NSL::Hubbard::Species::Hole);
    //    std::cout << std::setprecision(15) << Sf.hfm_.logDetM() << std::endl;
    
    clock_t tf = clock();

    std::cout << "# Routine ran in " << (float)(tf-ti) / CLOCKS_PER_SEC << " seconds" << std::endl;
    
    return EXIT_SUCCESS;
}
