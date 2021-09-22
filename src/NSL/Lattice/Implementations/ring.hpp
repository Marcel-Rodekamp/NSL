#ifndef NSL_LATTICE_RING_HPP
#define NSL_LATTICE_RING_HPP

#include "../lattice.hpp"

namespace NSL {
namespace Lattice {

template <typename Type>
class Ring: public NSL::Lattice::SpatialLatticeBase<Type> {
    public:
        Ring(const unsigned int n);

    private:
        unsigned int n_;

        static std::vector<Site> & sites_(const unsigned int n);
        static NSL::Tensor<Type> & hops_(const unsigned int n, const Type kappa=1.);
        static std::string & name_(const unsigned int n);
};

} // namespace Lattice
} // namespace NSL
#endif
