#include <chrono>
#include "NSL.hpp"
#include "highfive/H5File.hpp"
#include <yaml-cpp/yaml.h>

template<NSL::Concept::isNumber Type, typename LatticeType>
void writeMeta(LatticeType lat, NSL::Parameter & params, NSL::H5IO & h5, std::string BASENODE);

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
    // std::cout << params["Nx"] << std::endl;

    // Now we want to log the found parameters
    // - key is a std::string name,beta,...
    // - value is a ParameterEntry * which is a wrapper around the actual 
    //   value of interest, we can use ParameterEntry::repr() to get a string
    //   representation of the stored value
    // for(auto [key, value]: params){
    //     // skip these keys as they are logged in init already
    //     if (key == "device" || key == "file") {continue;}
    //     NSL::Logger::info( "{}: {}", key, value );
    // }

    // initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);
    if (lattice.sites() != params["Nx"].template to<NSL::size_t>()){
        throw std::runtime_error("The number of ions in the parameter file does not match the number of ions in the lattice.");
    }

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);

    NSL::Logger::info("Using beta={}, Nt={}, U={} on {}.", 
        params["beta"],
        params["Nt"],
        params["U"], 
        params["device"]
    );


    NSL::Action::HubbardFermionAction<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>
    > S_fermion(lattice,params);

    


    // Initialize the action being the sum of the gauge action & fermion action
    NSL::Action::Action S = S_fermion;

    // Initialize a configuration as starting point for the MC change
    NSL::Configuration<Type> config{
        {"phi", 
            NSL::Tensor<Type>(
                NSL::Device(params["device"]),
                NSL::size_t(params["Nt"]),
                NSL::size_t(params["Nx"])
            )
        }
    };

    //! \todo: we really need a proper random interface...
    config["phi"].randn();
    config["phi"] *= NSL::Hubbard::tilde<Type>(params, "U");
    config["phi"].imag() = NSL::RealTypeOf<Type>(0.0);


    Type act_val_pre = S_fermion.eval(config["phi"]);
    int N = 1000;
    double delta = 1./static_cast<double>(N);
    
    NSL::Configuration<Type> grad_val;
    NSL::Configuration<Type> grad2;


    // grad_val = S_fermion.bmmgrad(config["phi"]);
    // grad2 = S_fermion.grad(config["phi"]);
    // int loop_N = static_cast<int>(params["Nt"])*static_cast<int>(params["Nx"]);

    // for (int i=0; i<loop_N; i++) {
    //     std::cout << config["phi"][i] << " ; " << grad_val["phi"][i] << " ; " << grad2["phi"][i] << std::endl;
    // }


    // Gradient test
    std::cout << "Forward bmm evaluation" << std::endl;
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i=0; i < N; i++){
        NSL::Configuration<Type> grad_val = S_fermion.bmmgrad(config["phi"]);
        config["phi"] += delta;
    }
    std::cout << "Backward bmm evaluation" << std::endl;
    for (int i=0; i < N; i++){
        NSL::Configuration<Type> grad_val = S_fermion.bmmgrad(config["phi"]);
        config["phi"] -= delta;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_bmm = t4-t3;
    double t_bmm_val = t_bmm.count();
    std::cout << "Time: " << t_bmm.count() << std::endl;

    std::cout << "Forward standard evaluation" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i < N; i++){
        NSL::Configuration<Type> grad2 = S_fermion.grad(config["phi"]);
        config["phi"] += delta;
    }
    std::cout << "Backward standard evaluation" << std::endl;
    for (int i=0; i < N; i++){
        NSL::Configuration<Type> grad2 = S_fermion.grad(config["phi"]);
        config["phi"] -= delta;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t = t2 - t1;
    double t_val = t.count();
    std::cout << "Time: " << t.count() << std::endl;
    double speed_up = t_val/t_bmm_val;
    std::cout << "Speed up: " << speed_up << std::endl;
    // std::cout << grad2["phi"][0] << std::endl;
    // std::cout << grad_val["phi"][0] << std::endl;
    // // logdetM test
    // config["phi"].randn();

    // std::cout << "Forward bmm evaluation" << std::endl;
    // auto t3 = std::chrono::high_resolution_clock::now();
    // for (int i=0; i < N; i++){
    //     Type act_val2 = S_fermion.bmmeval(config["phi"]);
    //     config["phi"] += delta;
    // }
    // std::cout << "Backward bmm evaluation" << std::endl;
    // for (int i=0; i < N; i++){
    //     Type act_val2 = S_fermion.bmmeval(config["phi"]);
    //     config["phi"] -= delta;
    // }
    // auto t4 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> t_bmm = t4-t3;
    // std::cout << "Time: " << t_bmm.count() << std::endl;

    // std::cout << "Forward standard evaluation" << std::endl;
    // auto t1 = std::chrono::high_resolution_clock::now();
    // for (int i=0; i < N; i++){
    //     Type act_val2 = S_fermion.eval(config["phi"]);
    //     config["phi"] += delta;
    // }
    // std::cout << "Backward standard evaluation" << std::endl;
    // for (int i=0; i < N; i++){
    //     Type act_val2 = S_fermion.eval(config["phi"]);
    //     config["phi"] -= delta;
    // }
    // auto t2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> t = t2 - t1;
    // std::cout << "Time: " << t.count() << std::endl;

    // auto t3 = std::chrono::high_resolution_clock::now();
    // Type act_val2 = S_fermion.bmmeval(config["phi"]);
    // auto t4 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> t_bmm = t4-t3;
    // std::cout << "logdetM=" << act_val2 << std::endl;
    // std::cout << "Time: " << t_bmm.count() << std::endl;

    // auto t1 = std::chrono::high_resolution_clock::now();
    // Type act_val = S_fermion.eval(config["phi"]);
    // auto t2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> t = t2 - t1;
    // std::cout << "logdetM=" << act_val << std::endl;
    // std::cout << "Time: " << t.count() << std::endl;


    
    // // Initialize the integrator defining the equation of motion via the 
    // // real part of the force (real part of the action)
    // NSL::Integrator::LeapfrogRealForce leapfrog( 
    //     /*action*/S,  
    //     /*trajectoryLength*/NSL::RealTypeOf<Type>(params["trajectory length"]),
    //     /**numberSteps*/params["Nmd"]
    // );

    // // Initialize the RadialHMC
    // NSL::MCMC::RadialHMC hmc(leapfrog, S, h5);
    // NSL::Logger::stop_profile(init_time);

    // auto therm_time =  NSL::Logger::start_profile("Thermalization");
    // NSL::MCMC::MarkovState<Type> start_state;

    // NSL::Logger::info("Thermalizing {} steps...", params["Ntherm"].to<NSL::size_t>());
    // start_state = hmc.generate<NSL::MCMC::Chain::LastState>(config, params["Ntherm"].to<NSL::size_t>(), params["Nradial"], params["Nhmc"], params["radial scale"]);

    // NSL::Logger::stop_profile(therm_time);
    return EXIT_SUCCESS;
}