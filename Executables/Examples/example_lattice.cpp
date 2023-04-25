#include <chrono>
#include "NSL.hpp"

int main(int argc, char* argv[]){

    NSL::Logger::init_logger(argc, argv);
    
    auto init_time =  NSL::Logger::start_profile("Program Initialization");
    // Define the parameters of your system (you can also read these in...)
    typedef NSL::complex<double> Type;

    //    Number of ions (spatial sites)
    NSL::size_t Nx =  4;

    // Define the lattice geometry of interest
    // ToDo: Required for more sophisticated actions
    NSL::Lattice::Ring<Type> lattice(Nx, 1.0); 

    std::cout << lattice.hopping_matrix(1.0) << std::endl;
    std::cout << std::endl;
    auto [ee, ev]  = NSL::LinAlg::eigh(lattice.hopping_matrix(1.0));

    // ee gives the list of eigenvalues
    // if ee[i] gives the ith eigenvalue, then ev[:,i] is the corresponding eigenvector

    for (auto x: ev.shape()){
      std::cout << x << std::endl;
    }
    for (int i=0; i< Nx; i++) {
      for (int j=0; j< Nx; j++) {
	std::cout << NSL::LinAlg::inner_product(ev(NSL::Slice(), i),ev(NSL::Slice(),j)) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << NSL::LinAlg::mat_mul(ev.T(),ev) << std::endl;

    //    std::cout << ev << std::endl;
    
    return EXIT_SUCCESS;
}
