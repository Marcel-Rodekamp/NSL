#include "Action/Implementations/hubbardGaugeAction.tpp"
#include "Action/Implementations/hubbardFermiAction.tpp"
#include "Configuration/Configuration.tpp"
#include "Integrator/Impl/leapfrog.tpp"
#include "Integrator/Impl/rungeKutta4.tpp"
#include "NSL.hpp"
#include "complex.hpp"

int main(int argc, char ** argv){
    
    typedef NSL::complex<double> Type;
	
    // Initialize NSL
    NSL::Parameter params = NSL::init(argc, argv, "Example MCMC");
    // an example parameter file is can be found in example_param.yml
    
    auto init_time = NSL::Logger::start_profile("Initialization");
    
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
    // The chemical potential
    params.addParameter<Type>(
        "mu", yml["system"]["mu"].as<double>()
    );
    // The h5 file name to store the simulation results
    params.addParameter<std::string>(
        "h5file", yml["fileIO"]["h5file"].as<std::string>()
    );
    
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
    NSL::H5IO h5(params["h5file"].to<std::string>(),NSL::File::Truncate);

    // define the basenode for the h5file, everything is stored in 
    // params["h5Filename"]/BASENODE/
    std::string BASENODE;

    // initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);
    NSL::size_t dim = lattice.sites();

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);

    params.addParameter<decltype(lattice)>("lattice", lattice);

    NSL::Action::HubbardGaugeAction<Type> S_gauge(params);
    NSL::Action::HubbardFermionAction<
        Type,
        decltype(lattice),
        NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>
    > S_fermion(params);

    NSL::Action::Action S = S_gauge; // + S_fermion;

    NSL::Configuration<Type> q{{"phi",
        NSL::Tensor<Type>( params["device"].to<NSL::Device>(), 
                           params["Nt"].to<NSL::size_t>(), 
                           params["Nx"].to<NSL::size_t>() 
        )
    }};

    q["phi"].randn();

    NSL::RealTypeOf<Type> T = 0.1;

    NSL::Logger::info("Error \t step size");
    NSL::size_t numSteps = 1;
    //for(NSL::size_t numSteps = 10; numSteps < 210; numSteps+=10){
        NSL::Integrator::RungeKutta4 RK4(
            S, T, numSteps,  true
        );
        
        NSL::Configuration<Type> qt = RK4( q );

        NSL::Tensor<Type> dq = (qt["phi"] - q["phi"])/ T;

        NSL::Configuration<Type> F = S.grad(qt);

        auto diff = dq - F["phi"].conj();
        auto error = NSL::real(NSL::LinAlg::abs( diff )).sum();
        
        std::cout << diff.real()  << std::endl;
        std::cout << diff.imag()  << std::endl;

        NSL::Logger::info("{} \t {}", error, T/numSteps);
    //}

    return EXIT_SUCCESS;
}
