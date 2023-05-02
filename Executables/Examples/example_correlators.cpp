#include <chrono>
#include "NSL.hpp"

int main(int argc, char* argv[]){

    NSL::Logger::init_logger(argc, argv);

    // this routine requires an already generated ensemble, generated, for example, from example_MCMC
    std::string H5NAME("./1site_ring.v2.h5");  // name of h5 file with configurations
    std::string NODE("U2B4Nt32/markovChain");
    NSL::H5IO h5(H5NAME);
    
    auto init_time =  NSL::Logger::start_profile("Program Initialization");
    // Define the parameters of your system (you can also read these in...)
    typedef NSL::complex<double> Type;

    // if using an ensemble generated from example_MCMC, make sure tha the parameters below are the same
    
    // Number of ions (spatial sites)
    NSL::size_t Nx = 1;

    // Number of time slices
    NSL::size_t Nt = 32;

    // inverse temperature
    Type beta = 4.0;

    // Define the lattice geometry of interest
    // ToDo: Required for more sophisticated actions
    NSL::Lattice::Ring<Type> lattice(Nx); 

    std::cout << lattice.hopping_matrix(1.0) << std::endl;
    std::cout << std::endl;
    auto [e, u]  = lattice.eigh_hopping(1.0); // this routine returns the eigenenergies and eigenvectors of the hopping matrix
    // we need u to do the momentum projection

    // now let's try to calculate some correlators
    NSL::Tensor<Type> corr(Nt,Nx,Nx); // we calculate the Nx X Nx set of correlators
    
    // we need a container for the field phi
    NSL::Tensor<Type> phi(Nt,Nx);
    NSL::Configuration<Type> config{{"phi", phi}};

    // let's first do the non-interacting case
    // non-interacting means that all values of phi are zero:
    config["phi"].real()=0;
    config["phi"].imag()=0;

    int tsource;

    //now let's make a fermion (exp) matrix
    NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)> Mni(lattice, config["phi"], beta );

    // let's set up the CG
    NSL::LinAlg::CG<Type> invMMdni(Mni, NSL::FermionMatrix::MMdagger);

    // and allocate the solution vector
    NSL::Tensor<Type> b(Nt,Nx);

    
    tsource=0; // for the non-interacting case, one time source is sufficient, and we choose t=0

    for(int ni=0;ni<Nx;ni++){
       b.real()=0;
       b.imag()=0;
       b(tsource,NSL::Slice()) = u(ni,NSL::Slice());
    
       auto invMMdb = invMMdni(b);
       auto invMb = Mni.Mdagger(invMMdb);

       for (int nj=0;nj<Nx;nj++){
	 for (int t=0;t<Nt;t++){
	   corr(t,nj,ni)=NSL::LinAlg::inner_product(u(nj,NSL::Slice()),invMb(t,NSL::Slice()));
	 }
       }
    }

    for (int t=0;t<Nt;t++) {
      std::cout << "# t= "<< t <<  std::endl;
      for (int ni=0;ni<Nx;ni++) {
	for (int nj=0;nj<Nx;nj++)
	  std::cout << std::setprecision(15) << corr(t,ni,nj).real() << "\t";
	std::cout << std::endl;
      }
      std::cout << std::endl;
    }

    // now write this result out
    h5.write(corr,NODE+"/0/correlators/single/particle");
    
    
    // now let's read in a markov state
    NSL::MCMC::MarkovState<Type> markovstate;
    markovstate.configuration = config;

    for (int cfg = 1010; cfg<=200990; cfg+=10) {
      h5.read(markovstate,NODE,cfg);

      // initialize fermion matrix with this field
      NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)> M(lattice, markovstate.configuration["phi"], beta );

      // and also initialize the CG with this fermion matrix
      NSL::LinAlg::CG<Type> invMMd(M, NSL::FermionMatrix::MMdagger);

      // intialize the correlators
      corr.real()=0;
      corr.imag()=0;

      // in this case, we will loop over all possible time sources to increase statistics
      for (tsource=0;tsource<Nt;tsource++){
	for(int ni=0;ni<Nx;ni++){
	  b.real()=0;
	  b.imag()=0;
	  b(tsource,NSL::Slice()) = u(ni,NSL::Slice());
    
	  auto invMMdb = invMMd(b);
	  auto invMb = M.Mdagger(invMMdb);
	  // I don't know how to use the shift function
	  //	invMb = NSL::LinAlg::shift(invMb,-tsource,-1);
	  for (int nj=0;nj<Nx;nj++){
	    for (int t=0;t<Nt;t++){
	      if (t+tsource < Nt) {
		corr(t,nj,ni) += NSL::LinAlg::inner_product(u(nj,NSL::Slice()), invMb(t+tsource,NSL::Slice()))/Nt;
	      } else { // anti-periodic boundary conditions
		corr(t,nj,ni) -= NSL::LinAlg::inner_product(u(nj,NSL::Slice()), invMb(t+tsource-Nt,NSL::Slice()))/Nt;
	      }
	    }
	  }
	}
      }

      h5.write(corr,NODE+"/"+std::to_string(cfg)+"/correlators/single/particle");

      std::cout << "# cfg = " << cfg << std::endl;
      for (int t=0;t<Nt;t++) {
	std::cout <<  t <<  "\t";
	for (int ni=0;ni<Nx;ni++)
	  std::cout << std::setprecision(15) << corr(t,ni,ni).real() << "\t";
	std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    
    return EXIT_SUCCESS;
}

