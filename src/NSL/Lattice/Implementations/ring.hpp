#ifndef NSL_LATTICE_RING_HPP
#define NSL_LATTICE_RING_HPP

#include "../lattice.hpp"

namespace NSL::Lattice {

template <typename Type>
class Ring: public NSL::Lattice::SpatialLatticeBase<Type> {
    public:
        explicit Ring(const std::size_t n, const Type & kappa = 1);
};

} // namespace NSL
#endif
