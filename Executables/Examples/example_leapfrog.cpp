#include "Action/Implementations/hubbardGaugeAction.tpp"
#include "Action/Implementations/hubbardFermiAction.tpp"
#include "Integrator/Impl/leapfrog.tpp"
#include "NSL.hpp"

int main(int argc, char** argv){
    
    typedef NSL::complex<double> cd;
	
    NSL::Parameter params = NSL::init(argc, argv, "Testing the LeapFrog convergence with pseudofermions");

   // Initialize NSL
    
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

    NSL::size_t Nx =  params["Nx"];
    NSL::size_t Nt =  params["Nt"];
    NSL::Tensor<cd> phi(Nt,Nx); phi.randn(); 
    NSL::Tensor<cd> pi(Nt,Nx); pi.randn();

    phi.imag() = 0;
    pi.imag() = 0;
   
    // define configuration
    NSL::Configuration<cd> config{
		{"phi",phi}, 
    };

    // initialize the lattice 
    NSL::Lattice::Generic<cd> lattice(yml);
    if (lattice.sites() != params["Nx"].template to<NSL::size_t>()){
        throw std::runtime_error("The number of ions in the parameter file does not match the number of ions in the lattice.");
    }

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);

    // define momentum
    NSL::Configuration<cd> momentum{
		{"phi",pi}, 
	};

    NSL::Action::HubbardGaugeAction<cd> S_gauge(params);
    NSL::Action::PseudoFermionAction<cd,decltype(lattice), NSL::FermionMatrix::HubbardExp<cd,decltype(lattice)>>S_fermion(lattice, params);
    //NSL::Action::HubbardFermionAction<cd,decltype(lattice),NSL::FermionMatrix::HubbardExp<cd,decltype(lattice)>> S_fermion(lattice, params);
    // define the action
    NSL::Action::Action S = S_gauge + S_fermion;

    // compute pseudo fermions (if they don't exist this call does nothing)
    S.computePseudoFermion(config);
    
    cd Hi, Hf;

    Hi = (momentum["phi"] * momentum["phi"]).sum()/2.0 + S(config);

    for (int Nmd = 10; Nmd < 210; Nmd += 10){
      // define integrator
      NSL::Integrator::Leapfrog LF(
        /*action=*/ S,
        /*trajectoryLength=*/ .1,
        /*numberSteps=*/ Nmd,
        /*backward*/ false // optional
      );

      // integrate eom
      auto [config_proposal,momentum_proposal] = LF(/*q=*/config,/*p*/ momentum);
 
      Hf = (momentum_proposal["phi"] * momentum_proposal["phi"]).sum()/2.0 + S(config_proposal);
      std::cout << Nmd << "\t" << NSL::LinAlg::abs((Hf-Hi).real()/Hi.real()) << std::endl;
    }

    return EXIT_SUCCESS;
}
