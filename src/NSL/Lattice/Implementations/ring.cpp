#ifndef NSL_LATTICE_RING_CPP
#define NSL_LATTICE_RING_CPP

#include "../../Tensor/tensor.hpp"
#include "../lattice.hpp"
#include "ring.hpp"

namespace NSL::Lattice {

template<typename Type>
NSL::Lattice::Ring<Type>::Ring(const std::size_t n, const Type &kappa, const double &radius) :
        NSL::Lattice::SpatialLattice<Type>(
                "Ring(" + std::to_string(n) + ")",
                NSL::Tensor<Type>(n, n),
                std::vector<NSL::Lattice::Site>(n)
        ) {
    //! \todo: use a better pi
    double theta = 2*3.14159265358979 / n;
    for (int i = 0; i < n; ++i) {
        NSL::Tensor<double> coordinates(3);
        coordinates(0) = radius * cos(i * theta);
        coordinates(1) = radius * sin(i * theta);
        coordinates(2) = 0.;

        this->sites_[i].coordinates = coordinates;
    }
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
