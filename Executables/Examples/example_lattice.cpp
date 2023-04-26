#include <chrono>
#include "NSL.hpp"

int main(int argc, char* argv[]){

    NSL::Logger::init_logger(argc, argv);

    std::string H5NAME("./ensembles.h5");  // name of h5 file with configurations
    std::string NODE("2site/markovChain");
    NSL::H5IO h5(H5NAME);
    
    auto init_time =  NSL::Logger::start_profile("Program Initialization");
    // Define the parameters of your system (you can also read these in...)
    typedef NSL::complex<double> Type;

    //    Number of ions (spatial sites)
    NSL::size_t Nx =  2;

    NSL::size_t Nt = 32;

    // inverse temperature
    Type beta = 10.0;

    // Define the lattice geometry of interest
    // ToDo: Required for more sophisticated actions
    NSL::Lattice::Ring<Type> lattice(Nx); 

    std::cout << lattice.hopping_matrix(1.0) << std::endl;
    std::cout << std::endl;
    auto [e, u]  = lattice.eigh_hopping(1.0); // this routine returns the eigenenergies and eigenvectors of the hopping matrix

    // e gives the list of eigenvalues
    // if e[i] gives the ith eigenvalue, then u[i,:] is the corresponding eigenvector

    // now apply u kappa u^T to diagonalize the matrix
    //                                                  u                                kappa                                 u^T
    //std::cout << NSL::LinAlg::diag(NSL::LinAlg::mat_mul(u,NSL::LinAlg::mat_mul(lattice.hopping_matrix(1.0),NSL::LinAlg::transpose(u)))) << std::endl;

    //std::cout << std::endl;
    
    // now compare with the originally determined eigenvalues e
    std::cout << e << std::endl;

    // now let's read in a markov state
    NSL::MCMC::MarkovState<Type> markovstate;
    NSL::Tensor<Type> phi;
    NSL::Configuration<Type> config{{"phi", phi}};
    markovstate.configuration = config;
    
    h5.read(markovstate,NODE);

    markovstate.configuration["phi"].real()=0;
    markovstate.configuration["phi"].imag()=0;

    // now let's make a fermion (exp) matrix
    NSL::FermionMatrix::HubbardExp  M(lattice, markovstate.configuration["phi"], beta );

    //    std::cout << markovstate.configuration["phi"].real() << std::endl;

    // now set up the CG
    NSL::LinAlg::CG<Type> invMMd(M, NSL::FermionMatrix::MMdagger);

    // now make a solution vector
    NSL::Tensor<Type> b(Nt,Nx);

    NSL::Tensor<Type> corrp(Nt,Nx);
    int tsource=0;

    for(int ni=0;ni<Nx;ni++){
       b.real()=0;
       b.imag()=0;
       b(tsource,NSL::Slice()) = u(ni,NSL::Slice());
    
       auto invMMdb = invMMd(b);
       auto invMb = M.Mdagger(invMMdb);
  
       for (int t=0;t<Nt;t++){
	 corrp(t,ni)=NSL::LinAlg::inner_product(u(ni,NSL::Slice()),invMb(t,NSL::Slice()));
       }
    }

    for (int t=0;t<Nt;t++) {
      std::cout << t << " ";
      for (int ni=0;ni<Nx;ni++) 
	std::cout << std::setprecision(15) << corrp(t,ni).real() << "\t";
      std::cout << std::endl;
    }
    
    return EXIT_SUCCESS;
}
