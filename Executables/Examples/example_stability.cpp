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
    NSL::Action::HubbardGaugeAction<Type> Sg(params);

    // define a hubbard fermion action, the discretization (HubbardExp) is
    // hard wired in the meta data if you change this here, also change the
    // writeMeta()
    //
    NSL::Action::HubbardFermionAction<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>
      > Sf_direct(lattice,params);

    NSL::Action::HubbardFermionAction<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>
      > Sf_QR(lattice,params);

    NSL::Action::HubbardFermionAction<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>
      > Sf_SVD(lattice,params);

    Sf_QR.hfm_.stabilityMethod = "QR";// "QR", "DIRECTINVERSE"
    Sf_direct.hfm_.stabilityMethod = "DIRECTINVERSE";
    Sf_SVD.hfm_.stabilityMethod = "SVD";

    // Initialize the action being the sum of the gauge action & fermion action
    NSL::Action::Action S_direct = Sg + Sf_direct;
    NSL::Action::Action S_QR = Sg + Sf_QR;
    NSL::Action::Action S_SVD = Sg + Sf_SVD;

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
    // config["phi"] *= NSL::Hubbard::tilde<Type>(params, "U");
    config["phi"].imag() = 0.0;
    //config["phi"].real() = 0.0;

    //! \todo: we really need a proper random interface...
    momentum["phi"].randn();
    // momentum["phi"] *= NSL::Hubbard::tilde<Type>(params, "U");
    momentum["phi"].imag() = 0.0;

    S_direct(config);
    S_QR(config);
    S_SVD(config);

    Sf_direct.hfm_.populate(config["phi"], NSL::Hubbard::Species::Particle);
    Sf_QR.hfm_.populate(config["phi"], NSL::Hubbard::Species::Particle);
    Sf_SVD.hfm_.populate(config["phi"], NSL::Hubbard::Species::Particle);
    //std::cout << std::setprecision(15) << Sf_direct.hfm_.logDetM() << "\t" << Sf_SVD.hfm_.logDetM() << std::endl;
    std::cout << std::setprecision(15) << Sf_direct.hfm_.logDetM() << "\t" <<  Sf_QR.hfm_.logDetM() << "\t" << Sf_SVD.hfm_.logDetM() << std::endl;
    Sf_direct.hfm_.populate(config["phi"], NSL::Hubbard::Species::Hole);
    Sf_QR.hfm_.populate(config["phi"], NSL::Hubbard::Species::Hole);
    Sf_SVD.hfm_.populate(config["phi"], NSL::Hubbard::Species::Hole);
    std::cout << std::setprecision(15) << Sf_direct.hfm_.logDetM() << "\t" <<  Sf_QR.hfm_.logDetM() << "\t" << Sf_SVD.hfm_.logDetM() << std::endl;
    //std::cout << std::setprecision(15) << Sf_direct.hfm_.logDetM() << "\t" << Sf_SVD.hfm_.logDetM() << std::endl;

    double beta = params["beta"];
    std::cout << std::setprecision(15) << log(1.+exp(3.*beta))+log(1.+exp(1.*beta))+log(1.+exp(-1.*beta))+log(1.+exp(-3.*beta)) << std::endl;
    
    Type Hi_direct, Hf_direct;
    Type Hi_QR, Hf_QR;
    Type Hi_SVD, Hf_SVD;

    Hi_direct = (momentum["phi"] * momentum["phi"]).sum()/2.0 + S_direct(config);
    Hi_QR    = (momentum["phi"] * momentum["phi"]).sum()/2.0 + S_QR(config);
    Hi_SVD    = (momentum["phi"] * momentum["phi"]).sum()/2.0 + S_SVD(config);

    std::cout << "# H_/Nx :: " << std::setprecision(15) << Hi_direct/lattice.sites() << "\t" << Hi_QR/lattice.sites() << "\t" << Hi_SVD/lattice.sites() << std::endl; 
    
    for (int Nmd = 10; Nmd < 210; Nmd += 10){
      // define integrator
      NSL::Integrator::Leapfrog LF_direct(
       S_direct, // action
       1, // trajectoryLength
       Nmd, // numberSteps
       false // optional
      );
      NSL::Integrator::Leapfrog LF_QR(
         S_QR,
         1,
         Nmd,
         false // optional
      );
      NSL::Integrator::Leapfrog LF_SVD(
         S_SVD,
         1,
         Nmd,
         false // optional
      );

      // integrate eom
      auto [config_proposal,momentum_proposal] = LF_direct(config, momentum);
      auto [config_proposal2,momentum_proposal2] = LF_QR(config, momentum);
      auto [config_proposal3,momentum_proposal3] = LF_SVD(config, momentum);
 
      Hf_direct = (momentum_proposal["phi"] * momentum_proposal["phi"]).sum()/2.0 + S_direct(config_proposal);
      Hf_QR = (momentum_proposal2["phi"] * momentum_proposal2["phi"]).sum()/2.0 + S_QR(config_proposal2);
      Hf_SVD = (momentum_proposal3["phi"] * momentum_proposal3["phi"]).sum()/2.0 + S_SVD(config_proposal2);
      std::cout << Nmd << std::setprecision(15) << "\t" << NSL::LinAlg::abs((Hf_direct-Hi_direct).real()/Hi_direct.real()) << "\t"
		<< NSL::LinAlg::abs((Hf_QR-Hi_QR).real()/Hi_QR.real()) << "\t"
		<< NSL::LinAlg::abs((Hf_SVD-Hi_SVD).real()/Hi_SVD.real()) << std::endl;
    }
    

    return EXIT_SUCCESS;
}
