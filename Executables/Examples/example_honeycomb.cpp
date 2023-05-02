#include <chrono>
#include "NSL.hpp"

int main(int argc, char* argv[]){

    NSL::Logger::init_logger(argc, argv);

    // We'll loop over different combinations of L1 and L2 drawn from this set.
	//! \todo: This branch doesn't have HDF5 yet, so I'll just do one small example.
	// The hopping matrix of L=(3,3) is small enough to copy/paste and reformat by hand.
    std::set<int> LS = {3,};
	// When we can use HDF5 it'd be good to generate a variety of lattices, such as
    // std::set<int> LS = {3, 4, 5, 6, 9, 12, 15};
	double kappa = 1.0;

	// Now with a file named 
	//std::string filename = "./example_honeycomb.h5";
	// we create the HDF5 file

    for(auto L1: LS){
        for(auto L2: LS){

			std::vector<int> L = {L1, L2};

            NSL::Logger::info("Creating a honeycomb lattice with L1={}, L2={}...", L1, L2);
            NSL::Lattice::Honeycomb<double> lattice(L, kappa);
            NSL::Logger::info("... computing hopping matrix...");
			NSL::Tensor<double> hopping = lattice.hopping_matrix();

			// Here we write out the hopping matrix;
            std::cout << hopping << std::endl;

			// It'd be better to just
            //NSL::Logger::info("... writing to {} ...", filename);
			// and write to eg. /L1/L2/hopping
			// We could write the coordinates out too, for visualization.

        }
    }

    

    return EXIT_SUCCESS;
}
