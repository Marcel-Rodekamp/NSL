#include <chrono>
#include "NSL.hpp"
#include "highfive/H5File.hpp"
#include <yaml-cpp/yaml.h>

int main(int argc, char* argv[]){
    YAML::Node system;
    NSL::Logger::init_logger(argc, argv);

    system = YAML::LoadFile(argv[1]);

    std::string H5NAME(
       fmt::format("./"+system["h5file"].as<std::string>())
    );  // name of h5 file to store configurations, measurements, etc. . .
    NSL::H5IO h5(H5NAME);
    std::string BASENODE(
       fmt::format(system["name"].as<std::string>())
    );
    
    auto init_time =  NSL::Logger::start_profile("Program Initialization");
    // Define the parameters of your system (you can also read these in...)
    typedef NSL::complex<double> Type;

    //    Inverse temperature 
    Type beta = system["beta"].as<double>();
    
    //    On-Site Coupling
    Type U    = system["U"].as<double>();

    //    Number of time slices
    NSL::size_t Nt = system["nt"].as<int>();

    NSL::size_t saveFreq = system["checkpointing"].as<int>();
    
    // ToDo: Required for more sophisticated actions
    NSL::Lattice::Generic<Type> lattice(system);
    NSL::size_t dim = lattice.sites();

    //    std::cout << lattice.hopping_matrix(1.0) << std::endl;
    //    std::cout << std::endl;
    auto [e, u]  = lattice.eigh_hopping(1.0); // this routine returns the eigenenergies and eigenvectors of the hopping matrix

    // we need u to do the momentum projection

    // now let's try to calculate some correlators
    NSL::Tensor<Type> corr(Nt,dim,dim); // we calculate the Nx X Nx set of correlators
    
    // we need a container for the field phi
    NSL::Tensor<Type> phi(Nt,dim);
    NSL::Configuration<Type> config{{"phi", phi}};

    // let's first do the non-interacting case
    // non-interacting means that all values of phi are zero:
    config["phi"].real()=0;
    config["phi"].imag()=0;

    int tsource;

    //    NSL::size_t minCfg,maxCfg;
    auto [minCfg, maxCfg] = h5.getMinMaxConfigs(BASENODE+"/markovChain");
    
    //now let's make a fermion (exp) matrix
    NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)> Mni(lattice, config["phi"], beta );

    // let's set up the CG
    NSL::LinAlg::CG<Type> invMMdni(Mni, NSL::FermionMatrix::MMdagger);

    // and allocate the solution vector
    NSL::Tensor<Type> b(Nt,dim);
    
    tsource=0; // for the non-interacting case, one time source is sufficient, and we choose t=0

    for(int ni=0;ni<dim;ni++){
       b.real()=0;
       b.imag()=0;
       b(tsource,NSL::Slice()) = u(ni,NSL::Slice());
    
       auto invMMdb = invMMdni(b);
       auto invMb = Mni.Mdagger(invMMdb);

       for (int nj=ni;nj<ni+1;nj++){
	 for (int t=0;t<Nt;t++){
	   corr(t,nj,ni)=NSL::LinAlg::inner_product(u(nj,NSL::Slice()),invMb(t,NSL::Slice()));
	 }
       }
    }

    // now write this result out
    h5.write(corr,BASENODE+"/markovChain/0/correlators/single/particle");
    
    // now let's read in a markov state
    NSL::MCMC::MarkovState<Type> markovstate;
    markovstate.configuration = config;

    std::cout << "Min/Max configs are " << minCfg << "/" << maxCfg << std::endl;
    for (int cfg = minCfg; cfg<=maxCfg; cfg+=saveFreq) {
      h5.read(markovstate,BASENODE+"/markovChain",cfg);

      // initialize fermion matrix with this field
      NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)> M(lattice, markovstate.configuration["phi"], beta );

      // and also initialize the CG with this fermion matrix
      NSL::LinAlg::CG<Type> invMMd(M, NSL::FermionMatrix::MMdagger);

      // intialize the correlators
      corr.real()=0;
      corr.imag()=0;

      // in this case, we will loop over all possible time sources to increase statistics
      for (tsource=0;tsource<Nt;tsource += Nt){
	for(int ni=0;ni<dim;ni++){
	  b.real()=0;
	  b.imag()=0;
	  b(tsource,NSL::Slice()) = u(ni,NSL::Slice());
    
	  auto invMMdb = invMMd(b);
	  auto invMb = M.Mdagger(invMMdb);
	  // I don't know how to use the shift function
	  //	invMb = NSL::LinAlg::shift(invMb,-tsource,-1);
	  for (int nj=ni;nj<ni+1;nj++){
	    for (int t=0;t<Nt; t++){
	      if (t+tsource < Nt) {
		corr(t,nj,ni) += NSL::LinAlg::inner_product(u(nj,NSL::Slice()), invMb(t+tsource,NSL::Slice())); ///Nt;
	      } else { // anti-periodic boundary conditions
		corr(t,nj,ni) -= NSL::LinAlg::inner_product(u(nj,NSL::Slice()), invMb(t+tsource-Nt,NSL::Slice())); ///Nt;
	      }
	    }
	  }
	}
      }

      h5.write(corr,BASENODE+"/markovChain/"+std::to_string(cfg)+"/correlators/single/particle");
    }
    std::cout << "Min/Max configs are " << minCfg << "/" << maxCfg << std::endl;
    
    return EXIT_SUCCESS;
}

