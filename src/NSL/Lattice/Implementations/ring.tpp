#ifndef NSL_LATTICE_RING_TPP
#define NSL_LATTICE_RING_TPP

#include "ring.hpp"

namespace NSL::Lattice {

template<typename Type>
NSL::Lattice::Ring<Type>::Ring(const std::size_t n, const Type &kappa, const double &radius) :
        NSL::Lattice::SpatialLattice<Type>(
                "Ring(" + std::to_string(n) + ")",
                NSL::Tensor<Type>(n, n),
                NSL::Tensor<double>(n,3)
        )
{
    double theta = 2 * std::numbers::pi / n;

    // Sites are located around a circle of fixed radius.
    for(int i = 0; i < n; ++i) {
        this->sites_(i,0) = radius * std::cos(i * theta);
        this->sites_(i,1) = radius * std::sin(i * theta);
        this->sites_(i,2) = 0.;
    }

    for (int i = 0; i < n - 1; ++i) {
        this->hops_(i, i + 1) = kappa;
    }

    for (int i = 1; i < n; ++i) {
        this->hops_(i , i - 1) = NSL::LinAlg::conj(kappa);
    }

    // Periodic boundary conditions
    this->hops_(0, n - 1) = NSL::LinAlg::conj(kappa);
    this->hops_(n - 1, 0) = kappa;

    this->compute_adjacency();
}

} // namespace NSL::Lattice

#endif
