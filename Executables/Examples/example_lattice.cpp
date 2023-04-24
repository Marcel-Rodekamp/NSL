#include <chrono>
#include "NSL.hpp"

int main(int argc, char* argv[]){

    NSL::Logger::init_logger(argc, argv);
    
    auto init_time =  NSL::Logger::start_profile("Program Initialization");
    // Define the parameters of your system (you can also read these in...)
    typedef NSL::complex<double> Type;

    //    Number of ions (spatial sites)
    NSL::size_t Nx =  8;

    // Define the lattice geometry of interest
    // ToDo: Required for more sophisticated actions
    NSL::Lattice::Ring<Type> lattice(Nx, 1.0); 

    std::cout << lattice.adjacency_matrix() << std::endl;
    std::cout << std::endl;
    auto [ee, ev]  = NSL::LinAlg::eigh(lattice.hopping_matrix(1.0));
    std::cout << ee << std::endl;
    std::cout << std::endl;
    std::cout << ev(0,NSL::Slice()) << std::endl;
    
    return EXIT_SUCCESS;
}
