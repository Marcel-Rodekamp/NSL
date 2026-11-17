#ifndef NSL_LATTICE_GENERIC_TPP
#define NSL_LATTICE_GENERIC_TPP


/*! \file generic.tpp
 *  \author Thomas Luu
 *  \date   May 2023
 *  \brief  Implementation of the generic lattice given its adjacency and ion locations.
*/

#include <cmath>

#include "generic.hpp"
#include "LinAlg/inner_product.tpp"

namespace {
    std::string nameGeneric(const YAML::Node &system){
        std::string result=system["system"]["name"].as<std::string>();
        return result;
    }

    //! Spatial dimension of the lattice: the number of components of each entry of
    //! `system: positions` (2 for planar molecules, 3 for systems extending in z)
    int spatialDimGeneric(const YAML::Node &system){
        const YAML::Node positions = system["system"]["positions"];
        const int nions = system["system"]["nions"].as<int>();
        if (!positions || !positions.IsSequence() || int(positions.size()) != nions) {
            throw std::runtime_error("NSL::Lattice::Generic: `system: positions` must list `nions` positions");
        }
        const int dim = positions[0].size();
        for (int ni = 0; ni < nions; ++ni) {
            if (int(positions[ni].size()) != dim || dim < 1) {
                throw std::runtime_error(
                    "NSL::Lattice::Generic: all entries of `system: positions` must have the same number of coordinates");
            }
        }
        return dim;
    }
}

namespace NSL::Lattice {

template<typename Type>
inline int NSL::Lattice::Generic<Type>::get_number_of_sites_(const YAML::Node &system){       
    return system["system"]["nions"].as<int>();
}

template<typename Type>
void NSL::Lattice::Generic<Type>::init_(const YAML::Node &system)
{

    // all coordinates of each position (x, y and, if given, z)
    const int dim = spatialDimGeneric(system);
    for(int ni = 0; ni < system["system"]["nions"].as<int>(); ++ni){
        for(int d = 0; d < dim; ++d){
            this->sites_(ni, d) = system["system"]["positions"][ni][d].as<double>();
        }
    }

    int n1,n2;
    if (system["system"]["hopping"].size() > 1) { 
       for (int i=0; i< system["system"]["adjacency"].size();i++){
         n1 = system["system"]["adjacency"][i][0].as<int>();
         n2 = system["system"]["adjacency"][i][1].as<int>();
         this->hops_(n1,n2) = kappa * system["system"]["hopping"][i].as<double>();
         this->hops_(n2,n1) = kappa * system["system"]["hopping"][i].as<double>();
       }
     } else {
       double hh = system["system"]["hopping"].as<double>();
       for (int i=0; i< system["system"]["adjacency"].size();i++){
         n1 = system["system"]["adjacency"][i][0].as<int>();
         n2 = system["system"]["adjacency"][i][1].as<int>();
         this->hops_(n1,n2) = kappa * hh;
         this->hops_(n2,n1) = kappa * hh;
       }
     }

    this->compute_adjacency();

}

//=====================================================================
// Constructors
//=====================================================================

template<typename Type>
NSL::Lattice::Generic<Type>::Generic(
            const YAML::Node system,
            const Type & kappa
            ):
	    NSL::Lattice::SpatialLattice<Type>(
                nameGeneric(system),
                NSL::Tensor<Type>(this->get_number_of_sites_(system), this->get_number_of_sites_(system)),
                NSL::Tensor<double>(this->get_number_of_sites_(system), spatialDimGeneric(system))  // (nions, spatial dimension)
        ),
        kappa(kappa)
{

    this->init_(system);
    // We need to precompute this as the function is not set up for GPU.
    // In case the lattice is moved to the GPU we better have this called already ones
    // It will not be called a second time then.
    this->compute_bipartite();
}

} // namespace NSL::Lattice
#endif //NSL_LATTICE_GENERIC_TPP
