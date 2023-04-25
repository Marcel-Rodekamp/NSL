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
    NSL::Lattice::Square<Type> lattice({Nx,Nx}); 

    std::cout << lattice.hopping_matrix(1.0) << std::endl;
    std::cout << std::endl;
    auto [e, u]  = lattice.eigh_lattice(); //NSL::LinAlg::eigh(lattice.hopping_matrix(1.0));

    // e gives the list of eigenvalues
    // if e[i] gives the ith eigenvalue, then u[i,:] is the corresponding eigenvector

    // now apply u kappa u^T to diagonalize the matrix    
    std::cout << NSL::LinAlg::diag(NSL::LinAlg::mat_mul(u,NSL::LinAlg::mat_mul(lattice.hopping_matrix(1.0),NSL::LinAlg::transpose(u)))) << std::endl;

    std::cout << std::endl;
    
    // now compare with the originally determined eigenvalues ee
    std::cout << e << std::endl;

    
    return EXIT_SUCCESS;
}
