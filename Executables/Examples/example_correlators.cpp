#include <chrono>
#include "NSL.hpp"
#include "highfive/H5File.hpp"

int main(int argc, char* argv[]){

    NSL::Logger::init_logger(argc, argv);

    // this routine requires an already generated ensemble, generated, for example, from example_MCMC
    std::string H5NAME("./Honeycomb_Nt32_L16_L26_U2.000000+0.000000i_B1.000000+0.000000i.h5");  // name of h5 file with configurations
    std::string BASENODE(""); //BASENODE("1site/U10B6Nt40");
    NSL::H5IO h5(H5NAME);
    
    auto init_time =  NSL::Logger::start_profile("Program Initialization");
    // Define the parameters of your system (you can also read these in...)
    typedef NSL::complex<double> Type;

    // if using an ensemble generated from example_MCMC, make sure tha the parameters below are the same

    /*
    // Number of ions (spatial sites)
    NSL::size_t Nx = 6;
    NSL::size_t Ny = 6;

    NSL::size_t dim = Nx*Ny*2;

    // Number of time slices
    NSL::size_t Nt = 32;

    // inverse temperature
    Type beta = 1.0;

    // Define the lattice geometry of interest
    // ToDo: Required for more sophisticated actions
    NSL::Lattice::Honeycomb<Type> lattice({Nx,Ny});
    */
    
    NSL::size_t dim,Nt,saveFreq;
    Type beta;
    
    // load parameters from h5 file
    HighFive::File h5file = h5.getFile();

    std::complex<NSL::RealTypeOf<Type>> temp;
    // beta
    HighFive::DataSet dataset = h5file.getDataSet(BASENODE+"/Meta/params/beta");
    dataset.read(temp);
    beta = temp;

    // Nt
    dataset = h5file.getDataSet(BASENODE+"/Meta/params/Nt");
    dataset.read(Nt);

    // dim
    dataset = h5file.getDataSet(BASENODE+"/Meta/params/spatialDim");
    dataset.read(dim);

    // saveFreq
    dataset = h5file.getDataSet(BASENODE+"/Meta/params/saveFreq");
    dataset.read(saveFreq);
    
    // one day we will be able to get information about the exact lattice from the h5 file, but for now we do an explicit declaration
    NSL::Lattice::Ring<Type> lattice(dim);
    
    
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
      for (tsource=0;tsource<Nt;tsource++){
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
		corr(t,nj,ni) += NSL::LinAlg::inner_product(u(nj,NSL::Slice()), invMb(t+tsource,NSL::Slice()))/Nt;
	      } else { // anti-periodic boundary conditions
		corr(t,nj,ni) -= NSL::LinAlg::inner_product(u(nj,NSL::Slice()), invMb(t+tsource-Nt,NSL::Slice()))/Nt;
	      }
	    }
	  }
	}
      }

      h5.write(corr,BASENODE+"/markovChain/"+std::to_string(cfg)+"/correlators/single/particle");
    }
    
    return EXIT_SUCCESS;
}

