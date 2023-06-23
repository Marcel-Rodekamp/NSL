#include <chrono>
#include "NSL.hpp"
#include "highfive/H5File.hpp"
#include <yaml-cpp/yaml.h>

int main(int argc, char* argv[]){
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
    
    // In principle we can have a chemical potential, for this example we
    // assume it is zero
    params.addParameter<Type>("mu");

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
    NSL::H5IO h5(params["h5file"].to<std::string>());

    // define the basenode for the h5file, everything is stored in 
    // params["h5Filename"]/BASENODE/
    std::string BASENODE(fmt::format("{}",params["name"].repr()));

    // initialize the lattice 
    NSL::Lattice::Generic<Type> lattice(yml);
    NSL::size_t dim = lattice.sites();

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);

    params.addParameter<decltype(lattice)>("lattice", lattice);

    // this routine returns the eigenenergies and eigenvectors of the hopping matrix
    auto [e, u]  = lattice.eigh_hopping(1.0); 
    // we need u to do the momentum projection

    // now let's try to calculate some correlators
    NSL::Tensor<Type> corr(params["Nt"].to<NSL::size_t>(),dim,dim); // we calculate the Nx X Nx set of correlators
    
    // we need a container for the field phi
    NSL::Tensor<Type> phi(params["Nt"].to<NSL::size_t>(),dim);
    NSL::Configuration<Type> config{{"phi", phi}};

    // let's first do the non-interacting case
    // non-interacting means that all values of phi are zero:
    config["phi"] = Type(0);

    NSL::size_t tsource;
    // get the range of configuration ids from the h5file
    auto [minCfg, maxCfg] = h5.getMinMaxConfigs(BASENODE+"/markovChain");
    
    //now let's make a fermion (exp) matrix
    NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)> M(params);
    // and populate it with the configurations
    M.populate(config["phi"], NSL::Hubbard::Species::Particle);

    // let's set up the CG
    NSL::LinAlg::CG<Type> invMMdni(M, NSL::FermionMatrix::MMdagger);

    // allocate the solution vector
    NSL::Tensor<Type> b(params["Nt"].to<NSL::size_t>(),dim);

    // for the non-interacting case, one time source is sufficient, and we choose t=0
    tsource=0; 

    for(NSL::size_t ni = 0; ni < dim; ni++){
        b = Type(0);
        
        // put a source on the tsources time slice
        // the source is the ni's eigenvector of the hopping matrix,
        // like a momentum projection  
        b(tsource,NSL::Slice()) = u(ni, NSL::Slice());
    
        // invert MM^dagger
        auto invMMdb = invMMdni(b);

        // back multiply M^dagger to obtain M^{-1}
        auto invMb = M.Mdagger(invMMdb);

        // right multiply with the eigenvector to get the <u|M^{inv}|u> 
        // element. (Fourier component of the correlation function)
        for (NSL::size_t nj = ni; nj < ni+1; nj++){
	        for (NSL::size_t t = 0; t < params["Nt"].to<NSL::size_t>(); t++){
	            corr(t,nj,ni)=NSL::LinAlg::inner_product(u(nj,NSL::Slice()),invMb(t,NSL::Slice()));
	        }
        }
    }

    // now write this result out
    h5.write(corr,BASENODE+"/markovChain/0/correlators/single/particle");
    
    // prepare a markov state to be read from the h5fie
    NSL::MCMC::MarkovState<Type> markovstate;
    markovstate.configuration = config;

    NSL::Logger::info("Min/Max configs are {}/{}",minCfg,maxCfg);
    for (NSL::size_t cfg = minCfg; cfg<=maxCfg; cfg+=NSL::size_t(params["save frequency"])) {
        h5.read(markovstate,BASENODE+"/markovChain",cfg);

        // populate fermion matrix with this field
        M.populate(markovstate.configuration["phi"], NSL::Hubbard::Species::Particle);

        // and also initialize the CG with this fermion matrix
        NSL::LinAlg::CG<Type> invMMd(M, NSL::FermionMatrix::MMdagger);

        // intialize the correlators
        corr = Type(0);

        // in this case, we will loop over all possible time sources to increase statistics
        for(tsource = 0; tsource < params["Nt"].to<NSL::size_t>(); tsource += params["Nt"].to<NSL::size_t>()){
	        for(NSL::size_t ni = 0; ni < dim; ni++){
                b = Type(0);
	            
                // again retrieve the eigenvector 
                // put a source on the tsources time slice
                // the source is the ni's eigenvector of the hopping matrix,
                // like a momentum projection
                b(tsource,NSL::Slice()) = u(ni,NSL::Slice());
    
                // invert MM^dagger
	            auto invMMdb = invMMd(b);
	            
                // back-multiply M^dagger to obtain M^{-1}
                auto invMb = M.Mdagger(invMMdb);

                // ToDo: This is not available for GPU calculations!
                // right multiply with the eigenvector to get the <u|M^{inv}|u> 
                // element. (Fourier component of the correlation function)
	            for (NSL::size_t nj = ni; nj < ni+1; nj++){
	                for (NSL::size_t t = 0; t < params["Nt"].to<NSL::size_t>(); t++){

	                    if (t+tsource < params["Nt"].to<NSL::size_t>()) {
		                    corr(t,nj,ni) += NSL::LinAlg::inner_product(
                                u(nj,NSL::Slice()), 
                                invMb(t+tsource,NSL::Slice())
                            ); 
	                    } else { 
                            // anti-periodic boundary conditions
		                    corr(t,nj,ni) -= NSL::LinAlg::inner_product(
                                u(nj,NSL::Slice()), 
                                invMb(t+tsource-params["Nt"].to<NSL::size_t>(), NSL::Slice())
                            );
	                    }
	                } // for t
	            } // for nj
	        } //for ni
        } // for tsource

        // write the resutl
        h5.write(corr,BASENODE+"/markovChain/"+std::to_string(cfg)+"/correlators/single/particle");
    } // for cfg
    
    return EXIT_SUCCESS;
}

