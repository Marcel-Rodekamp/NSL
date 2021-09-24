#ifndef NSL_LATTICE_RING_HPP
#define NSL_LATTICE_RING_HPP

#include "../lattice.hpp"

namespace NSL::Lattice {

/*! Sites evenly distributed around a circle.
 **/
template <typename Type>
class Ring: public NSL::Lattice::SpatialLattice<Type> {
    public:
        /*!
         *  \param n the number of sites
         *  \param kappa the hopping amplitude.
         *  \param radius 
         *  \todo  If kappa is complex, it describes hopping to the right.
         **/
        explicit Ring(const std::size_t n, const Type & kappa = 1, const double &radius = -1);
};

} // namespace NSL
#endif
