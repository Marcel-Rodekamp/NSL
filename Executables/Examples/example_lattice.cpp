#include <chrono>
#include "NSL.hpp"

int main(int argc, char* argv[]){

    NSL::Logger::init_logger(argc, argv);
    
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
    std::cout << "# this is u^T kappa u = diag(ee)" << std::endl;
    //                                                  u                                kappa                                 u^T
    std::cout << NSL::LinAlg::diag(NSL::LinAlg::mat_mul(u,NSL::LinAlg::mat_mul(lattice.hopping_matrix(1.0),NSL::LinAlg::transpose(u)))) << std::endl;

    std::cout << std::endl;
    
    // now compare with the originally determined eigenvalues e
    std::cout << "# these are the calculated eigenvalues" << std::endl;
    std::cout << e << std::endl;

    std::cout << std::endl;
    
    // this is the exponentiated hopping matrix
    std::cout << "# this is the exponentiated hopping matrix" << std::endl;
    std::cout << lattice.exp_hopping_matrix(1.0) << std::endl;

    return EXIT_SUCCESS;
}

