/*! \file complete.cpp
*/

#include "complete.hpp"

namespace NSL::Lattice {

template<typename Type>
NSL::Lattice::Complete<Type>::Complete(const std::size_t n, const Type &kappa, const double &radius) :
        NSL::Lattice::SpatialLattice<Type>(
                "K(" + std::to_string(n) + ")",
                NSL::Tensor<Type>(n, n),
                NSL::Tensor<double>(n,3)
        )
{
    //! todo
    //if(constexpr(is_complex(Type))){
    //    // fail.  raise an exception?
    //}

    double theta = 2 * std::numbers::pi / n;

    // Sites are located around a circle of fixed radius.
    for(int i = 0; i < n; ++i) {
        this->sites_(i,0) = radius * std::cos(i * theta);
        this->sites_(i,1) = radius * std::sin(i * theta);
        this->sites_(i,2) = 0.;
    }

    // Every site is connected to every other:
    this->hops_ = kappa;
    for (int i = 0; i < n ; ++i) {
        this->hops_(i, i) = static_cast<Type>(0);
    }

    this->compute_adjacency();

}

} // namespace NSL::Lattice
