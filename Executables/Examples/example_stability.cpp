#include "Action/Implementations/hubbardGaugeAction.tpp"
#include "Action/Implementations/hubbardFermiAction.tpp"
#include "NSL.hpp"
#include <chrono>

int main(int argc, char* argv[]){
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

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
    config["phi"] *= NSL::Hubbard::tilde<Type>(params, "U");
    if (yml["system"]["offset"]){
      config["phi"].imag() = NSL::RealTypeOf<Type>(params["offset"]);
    } else {
    config["phi"].imag() = 0.0;
    }
    
    //! \todo: we really need a proper random interface...
    momentum["phi"].randn();
    momentum["phi"].imag() = 0.0;

    S_direct(config);
    S_QR(config);
    S_SVD(config);

    std::cout << "# value of action S for DIRECTINVERSE \t QR \t SVD" << std::endl;
    Sf_direct.hfm_.populate(config["phi"], NSL::Hubbard::Species::Particle);
    Sf_QR.hfm_.populate(config["phi"], NSL::Hubbard::Species::Particle);
    Sf_SVD.hfm_.populate(config["phi"], NSL::Hubbard::Species::Particle);
    std::cout << std::setprecision(15) << Sf_direct.hfm_.logDetM() << "\t" <<  Sf_QR.hfm_.logDetM() << "\t" << Sf_SVD.hfm_.logDetM() << std::endl;
    Sf_direct.hfm_.populate(config["phi"], NSL::Hubbard::Species::Hole);
    Sf_QR.hfm_.populate(config["phi"], NSL::Hubbard::Species::Hole);
    Sf_SVD.hfm_.populate(config["phi"], NSL::Hubbard::Species::Hole);
    std::cout << std::setprecision(15) << Sf_direct.hfm_.logDetM() << "\t" <<  Sf_QR.hfm_.logDetM() << "\t" << Sf_SVD.hfm_.logDetM() << std::endl;

    Type Hi_direct, Hf_direct;
    Type Hi_QR, Hf_QR;
    Type Hi_SVD, Hf_SVD;

    Hi_direct = (momentum["phi"] * momentum["phi"]).sum()/2.0 + S_direct(config);
    Hi_QR    = (momentum["phi"] * momentum["phi"]).sum()/2.0 + S_QR(config);
    Hi_SVD    = (momentum["phi"] * momentum["phi"]).sum()/2.0 + S_SVD(config);

    std::cout << "# H_/Nx :: " << std::setprecision(15) << Hi_direct/lattice.sites() << "\t" << Hi_QR/lattice.sites() << "\t" << Hi_SVD/lattice.sites() << std::endl; 

    double U = params["U"];
    double beta = params["beta"];
    double trajLength = 3.14159265*sqrt(U*beta/Nt)/2;
    // std::cout << std::setprecision(15) << "traj. length = " << trajLength << std::endl;
    int Nmd = 1;
    std::cout << "Nmd = " << Nmd << std::endl;
    std::cout << "Lattice size = " << lattice.sites() << std::endl;
    std::cout << "Nt = " << Nt << std::endl;
    std::cout << "Total system size = " << lattice.sites() * Nt << std::endl;
    std::vector<float> times_direct, times_QR, times_SVD;
    for (int Nmd = 10; Nmd < 210; Nmd += 10){
      // define integrator
      NSL::Integrator::LeapfrogRealForce LF_direct(
       S_direct, // action
       trajLength, // trajectoryLength
       Nmd, // numberSteps
       false // optional
      );
      NSL::Integrator::LeapfrogRealForce LF_QR(
         S_QR,
         trajLength,
         Nmd,
         false // optional
      );
      NSL::Integrator::LeapfrogRealForce LF_SVD(
         S_SVD,
         trajLength,
         Nmd,
         false // optional
      );

      // integrate eom
    auto ti = high_resolution_clock::now();

    auto [config_proposal,momentum_proposal] = LF_direct(config, momentum);
    auto tf1 = high_resolution_clock::now();

    auto [config_proposal2,momentum_proposal2] = LF_QR(config, momentum);
    auto tf2 = high_resolution_clock::now();
      
    auto [config_proposal3,momentum_proposal3] = LF_SVD(config, momentum);
    auto tf3 = high_resolution_clock::now();

      Hf_direct = (momentum_proposal["phi"] * momentum_proposal["phi"]).sum()/2.0 + S_direct(config_proposal);
      Hf_QR = (momentum_proposal2["phi"] * momentum_proposal2["phi"]).sum()/2.0 + S_QR(config_proposal2);
      Hf_SVD = (momentum_proposal3["phi"] * momentum_proposal3["phi"]).sum()/2.0 + S_SVD(config_proposal3);
      std::cout << Nmd << std::setprecision(15) << "\t" << NSL::LinAlg::abs((Hf_direct-Hi_direct).real()) << "\t"
		<< NSL::LinAlg::abs((Hf_QR-Hi_QR).real()) << "\t"
		<< NSL::LinAlg::abs((Hf_SVD-Hi_SVD).real()) << std::endl;

    duration<double, std::milli> durr1 = tf1 - ti;
    duration<double, std::milli> durr2 = tf2 - tf1;
    duration<double, std::milli> durr3 = tf3 - tf2;
    times_direct.push_back(durr1.count());
    times_QR.push_back(durr2.count());
    times_SVD.push_back(durr3.count());

    std::cout << "# Routine ran in " << durr1.count() << " milliseconds for DIRECTINVERSE" << std::endl;
    std::cout << "# Routine ran in " << durr2.count() << " milliseconds for QR" << std::endl;
    std::cout << "# Routine ran in " << durr3.count() << " milliseconds for SVD" << std::endl;
    std::cout << " " << std::endl;
    }
    std::cout << "Average time per step: " << std::accumulate(times_direct.begin(), times_direct.end(), 0.0)/times_direct.size() << " ms for DIRECTINVERSE" << std::endl;
    std::cout << "Average time per step: " << std::accumulate(times_QR.begin(), times_QR.end(), 0.0)/times_QR.size() << " ms for QR" << std::endl;
    std::cout << "Ratio of average time per step QR / DIRECTINVERSE: " << std::accumulate(times_QR.begin(), times_QR.end(), 0.0)/times_QR.size() / (std::accumulate(times_direct.begin(), times_direct.end(), 0.0)/times_direct.size()) << std::endl;
    std::cout << "Average time per step: " << std::accumulate(times_SVD.begin(), times_SVD.end(), 0.0)/times_SVD.size() << " ms for SVD" << std::endl;
    std::cout << "Ratio of average time per step SVD / DIRECTINVERSE: " << std::accumulate(times_SVD.begin(), times_SVD.end(), 0.0)/times_SVD.size() / (std::accumulate(times_direct.begin(), times_direct.end(), 0.0)/times_direct.size()) << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;

    return EXIT_SUCCESS;
}
