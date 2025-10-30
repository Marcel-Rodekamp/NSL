#include "Action/Implementations/hubbardGaugeAction.tpp"
#include "Action/Implementations/hubbardFermiAction.tpp"
#include "Integrator/Impl/leapfrog.tpp"
#include "NSL.hpp"

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

    if (yml["stability"] ) {
      params["stability"]    = yml["stability"].as<std::string>();
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


    // initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);
    if (lattice.sites() != params["Nx"].template to<NSL::size_t>()){
        throw std::runtime_error("The number of ions in the parameter file does not match the number of ions in the lattice.");
    }

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);

    // define a hubbard gauge action
    NSL::Action::HubbardGaugeAction<Type> S_gauge(params);

    // define a hubbard fermion action, the discretization (HubbardExp) is
    // hard wired in the meta data if you change this here, also change the
    // writeMeta()
    //
    NSL::Action::HubbardFermionAction<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>
      > S_fermion(lattice,params);

    // uncomment the next line if you want to choose a specific stabilizer.  default is "QR"
    if (yml["stability"] ) {
      std::string stabilityMethod = params["stability"];
      S_fermion.hfm_.stabilityMethod = stabilityMethod;// "QR", "DIRECTINVERSE", "SVD"
    }

    // Initialize the action being the sum of the gauge action & fermion action
    NSL::Action::Action S = S_gauge + S_fermion;


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

    NSL::Configuration<Type> momentum{
        {"phi",
            NSL::Tensor<Type>(
                NSL::Device(params["device"]),
                NSL::size_t(params["Nt"]),
                NSL::size_t(params["Nx"])
            )
        }
    };

    NSL::setSeed(1234);

    //! \todo: we really need a proper random interface...
    config["phi"].randn();
    config["phi"] *= NSL::Hubbard::tilde<Type>(params, "U");
    config["phi"].imag() = 0.0;
    //config["phi"].real() = 0.0;

    //! \todo: we really need a proper random interface...
    momentum["phi"].randn();
    momentum["phi"].imag() = 0.0;

    Type Hi, Hf;
    double U = params["U"];
    double beta = params["beta"];
    double trajLength = 3.14159265*sqrt(U*beta/Nt)/2;
    std::cout << "traj. length = " << trajLength << std::endl;

    Hi = (momentum["phi"] * momentum["phi"]).sum()/2.0 + S(config);

    for (int Nmd = 10; Nmd < 210; Nmd += 10){
      // define integrator
      NSL::Integrator::Leapfrog LF(
        /*action=*/ S,
        /*trajectoryLength=*/ trajLength,
        /*numberSteps=*/ Nmd,
        /*backward*/ false // optional
      );

      // integrate eom
      auto [config_proposal,momentum_proposal] = LF(/*q=*/config,/*p*/ momentum);
 
      Hf = (momentum_proposal["phi"] * momentum_proposal["phi"]).sum()/2.0 + S(config_proposal);
      std::cout << Nmd << "\t (Hf,Hi) = (" << std::setprecision(15) << Hf <<", "<< Hi <<") dH = " << NSL::LinAlg::abs((Hf-Hi)) << std::endl;
    }

    return EXIT_SUCCESS;
}
