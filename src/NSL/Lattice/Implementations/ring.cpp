#ifndef NSL_LATTICE_RING_CPP
#define NSL_LATTICE_RING_CPP

#include "../../Tensor/tensor.hpp"
#include "../lattice.hpp"
#include "ring.hpp"

namespace NSL::Lattice {

template<typename Type>
NSL::Lattice::Ring<Type>::Ring(const std::size_t n, const Type &kappa) :
        NSL::Lattice::SpatialLattice<Type>(
                "Ring(" + std::to_string(n) + ")",
                NSL::Tensor<Type>(n, n),
                std::vector<NSL::Lattice::Site>(n)
        ) {
    for (int i = 0; i < n - 1; ++i) {
        this->hops_(i, i + 1) = kappa;
    }
    for (int i = 1; i < n; ++i) {
        this->hops_(i - 1, i) = kappa;
    }
    this->hops_(0, n - 1) = kappa;
    this->hops_(n - 1, 0) = kappa;
}

} // namespace NSL::Lattice

template class NSL::Lattice::Ring<float>;
template class NSL::Lattice::Ring<double>;

#endif
