#ifndef NSL_LATTICE_RING_HPP
#define NSL_LATTICE_RING_HPP

#include "../lattice.hpp"

namespace NSL::Lattice {

/*! A one-dimensional spatial lattice with periodic boundary conditions.
 *      Can be thought of as layed out in a line with ends identified,
 *      or distributed equally around a circle.
 *      Might need to disambiguate these if a coordinate-dependent potential is used.
 **/
template <typename Type>
class Ring: public NSL::Lattice::SpatialLattice<Type> {
    public:
        /*!
         *  \param n the number of sites
         *  \param kappa the hopping amplitude.
         *  \todo  If kappa is complex, it describes hopping to the right.
         **/
        explicit Ring(const std::size_t n, const Type & kappa = 1);
};

} // namespace NSL
#endif
