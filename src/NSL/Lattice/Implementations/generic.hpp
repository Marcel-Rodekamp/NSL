#ifndef NSL_LATTICE_GENERIC_HPP
#define NSL_LATTICE_GENERIC_HPP

#include "../lattice.hpp"
#include <yaml-cpp/yaml.h>

namespace NSL::Lattice {

/*! A generic lattice given its adjacency matrix and ion locations
 **/
template <typename Type>
class Generic: public NSL::Lattice::SpatialLattice<Type> {
    public:
        /*!
         *  \param L the number of sites in the a1 and a2 directions
         *  \param kappas the hopping parameter in each direction
         *
         **/
        explicit Generic(
                const YAML::Node system,
                const Type &kappa = 1.);

    protected:
        Type kappa;
    private:

        //  inline int unit_cells_(const std::vector<int> &L);
        inline int get_number_of_sites_(const YAML::Node &system);
        void init_(const YAML::Node &system);
};

} // namespace NSL::Lattice

#endif // NSL_LATTICE_GENERIC_HPP
