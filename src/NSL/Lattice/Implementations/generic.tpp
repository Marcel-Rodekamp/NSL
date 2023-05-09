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
        std::string result=system["name"].as<std::string>();
        return result;
    }
}

namespace NSL::Lattice {

template<typename Type>
inline int NSL::Lattice::Generic<Type>::get_number_of_sites_(const YAML::Node &system){       
    return system["nions"].as<int>();
}

template<typename Type>
void NSL::Lattice::Generic<Type>::init_(const YAML::Node &system)
{

    for(int ni = 0; ni < system["nions"].as<int>(); ++ni){
    	  this->sites_(ni, 0) = system["positions"][ni][0].as<double>();
	  this->sites_(ni, 1) = system["positions"][ni][1].as<double>();
    }

    int n1,n2;
    if (system["hopping"].size() > 1) { 
       for (int i=0; i< system["adjacency"].size();i++){
         n1 = system["adjacency"][i][0].as<int>();
         n2 = system["adjacency"][i][1].as<int>();
         this->hops_(n1,n2) = kappa * system["hopping"][i].as<double>();
         this->hops_(n2,n1) = kappa * system["hopping"][i].as<double>();
       }
     } else {
       double hh = system["hopping"].as<double>();
       for (int i=0; i< system["adjacency"].size();i++){
         n1 = system["adjacency"][i][0].as<int>();
         n2 = system["adjacency"][i][1].as<int>();
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
                NSL::Tensor<double>(this->get_number_of_sites_(system),2)  // the "2" is for the spatial dimension (I believe)
        ),
        kappa(kappa)
{

    this->init_(system);
}

} // namespace NSL::Lattice
#endif //NSL_LATTICE_GENERIC_TPP
