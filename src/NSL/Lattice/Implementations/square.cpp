#ifndef NSL_LATTICE_SQUARE_CPP
#define NSL_LATTICE_SQUARE_CPP

/*! \file square.cpp
*/

#include <cmath>

#include "../../Tensor/tensor.hpp"
#include "../lattice.hpp"
#include "square.hpp"


namespace NSL::Lattice {

template<typename Type>
inline std::size_t NSL::Lattice::Square<Type>::n_to_sites_(const std::vector<std::size_t> &n){
    std::size_t volume=1;
    for(const auto& value: n) volume *= value;
    return volume;
}

template<typename Type>
NSL::Tensor<int> NSL::Lattice::Square<Type>::integer_coordinates_(const std::vector<std::size_t> &n){
    std::size_t sites = NSL::Lattice::Square<Type>::n_to_sites_(n);
    std::size_t dimensions = n.size();
    NSL::Tensor<int> coordinates(sites, dimensions);

    std::size_t copies = 1;
    std::size_t run = sites;
    // We set up the list of sites so that the 0th dimension is slowest
    // and the last dimension is fastest.  If we knew ahead of time how
    // many dimensions we'd be given, that'd be quite simple: we'd just 
    // hard-code the loops.  Instead, with a generic implementation we
    // need to think more generally.
    //
    // We'll fill in each coordinate dimension-by-dimension.
    for(std::size_t d = 0; d < dimensions; ++d) {
    // In each dimension, the speed of the coordinate gets faster,
        run /= n[d];
    // and we require more copies---more loops over the values of the
    // coordinates.
        for(std::size_t copy = 0; copy < copies; ++copy){
    // For each copy, loop over the values of the coordinates
            for(std::size_t x = 0; x < n[d]; ++x) {
    // and repeat each coordinate some number of times.
                for(std::size_t repeat=0; repeat < run; ++repeat){
                    // There is a lot of magic in the first index.
                    // copy*run*n[d] fast-forwards past previous copies,
                    // x*run fast-forwards past smaller coordinates in this copy
                    // repeat iterates across all the repetitions
                    coordinates(copy*run*n[d]+x*run + repeat, d) = x;
                }
            }
        }
    // Finally, the next dimension goes faster, so we need more copies.
        copies *= n[d];
    }
    return coordinates;
}

template<typename Type>
NSL::Lattice::Square<Type>::Square(
            const std::vector<std::size_t> n,
            const Type & kappa,
            const double spacing
            ):
        NSL::Lattice::SpatialLattice<Type>(
                "Square()",    //! todo: stringify
                NSL::Tensor<Type>(this->n_to_sites_(n), this->n_to_sites_(n)),
                NSL::Tensor<double>(this->n_to_sites_(n), n.size())
        ),
        integers_(integer_coordinates_(n))
{
    std::vector<Type> kappas(n.size());
    for(std::size_t i=0; i < kappas.size(); ++i){
        kappas[i] = kappa;
    }

    std::vector<double> spacings(n.size());
    for(std::size_t i=0; i < spacings.size(); ++i){
        spacings[i] = spacing;
    }

    this->init_(n, kappas, spacings);
}

template<typename Type>
void NSL::Lattice::Square<Type>::init_(const std::vector<std::size_t> &n,
                   const std::vector<Type> &kappa,
                   const std::vector<double> spacings)
{
    for(int i = 0; i < this->sites(); ++i) {
        for(int d = 0; d < n.size(); ++d) {
            this->sites_(i,d) = spacings[d] * this->integers_(i,d);
        }
    }

    for (int i = 0; i < this->sites(); ++i){
        for (int j = i; j < this->sites(); ++j){
            //! todo: properly determine nearest-neighbors
            // depends on direction-dependent kappa
            this->hops_(i,j) = 0;
            this->hops_(j,i) = 0;
        }
    }

}

} // namespace NSL::Lattice

template class NSL::Lattice::Square<float>;
template class NSL::Lattice::Square<double>;

#endif
