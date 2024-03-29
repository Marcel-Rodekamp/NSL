#ifndef NSL_LATTICE_SQUARE_TPP
#define NSL_LATTICE_SQUARE_TPP


/*! \file square.cpp
 *  \author Evan Berkowitz
 *  \date October 2021
 *  \brief Implementation of arbitrary-dimension square lattice with different hopping amplitudes and lattice spacings.
*/

#include <cmath>

#include "square.hpp"

namespace {
    std::string name(const std::vector<std::size_t> &n){
        std::string result="Square(";
        for(const auto& value: n){
            result+= " ";
            result+= std::to_string(value);
        }
        result += " )";
        return result;
    }
}

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
void NSL::Lattice::Square<Type>::init_(const std::vector<std::size_t> &n,
                   const std::vector<Type> &kappas,
                   const std::vector<double> spacings)
{
    assertm(n.size() == kappas.size(), "hopping amplitudes and dimension mismatch");
    assertm(n.size() == spacings.size(), "lattice spacings and dimension mismatch");

    this->kappas_ = kappas;
    this->spacings_ = spacings;

    for(int i = 0; i < this->sites(); ++i) {
        for(int d = 0; d < n.size(); ++d) {
            this->sites_(i,d) = spacings[d] * this->integers_(i,d);
        }
    }

    //! todo this algorithm is quadratic in the number of sites.
    // I think, with intelligence, it can be made linear.
    // But, it works for now.
    for (int i = 0; i < this->sites(); ++i){
        for (int j = i; j < this->sites(); ++j){
            // This does NOT handle periodic boundary conditions.
            std::size_t same = 0;
            std::size_t adjacent = 0;
            std::size_t dim = -1;
            for(int d = 0; d < n.size(); ++d){
                int diff = this->integers_(j,d)-this->integers_(i,d);
                if( diff == 0 ) {same+=1;}
                if( diff == +1 || diff == -1) {adjacent+=1; dim=d;}
            }
            if(adjacent == 1 && same + 1 == n.size() ){
                // depends on direction-dependent kappas
                this->hops_(i,j) = kappas[dim];
                //! \todo The following should REALLY be conj(kappas[dim])
                // fixing this correctly may require tracking more carefully
                // whether diff == +1 or -1.
                // Moreover, there's little point in a fix until issue #9
                // https://github.com/Marcel-Rodekamp/NSL/issues/9
                // is resolved
                this->hops_(j,i) = kappas[dim];
            }
        }
    }

    this->compute_adjacency();

}

//=====================================================================
// Constructors
//=====================================================================

template<typename Type>
NSL::Lattice::Square<Type>::Square(
// All vector
            const std::vector<std::size_t> n,
            const std::vector<Type> & kappas,
            const std::vector<double> spacings
            ):
        NSL::Lattice::SpatialLattice<Type>(
                name(n),
                NSL::Tensor<Type>(this->n_to_sites_(n), this->n_to_sites_(n)),
                NSL::Tensor<double>(this->n_to_sites_(n), n.size())
        ),
        dimensions_(n),
        integers_(integer_coordinates_(n))
{
    this->init_(n, kappas, spacings);
}

template<typename Type>
NSL::Lattice::Square<Type>::Square(
// scalar spacing
            const std::vector<std::size_t> n,
            const std::vector<Type> &kappas,
            const double spacing
            ):
        NSL::Lattice::SpatialLattice<Type>(
                name(n),
                NSL::Tensor<Type>(this->n_to_sites_(n), this->n_to_sites_(n)),
                NSL::Tensor<double>(this->n_to_sites_(n), n.size())
        ),
        dimensions_(n),
        integers_(integer_coordinates_(n))
{
    std::vector<double> spacings(n.size());
    for(std::size_t i=0; i < spacings.size(); ++i){
        spacings[i] = spacing;
    }

    this->init_(n, kappas, spacings);
}

template<typename Type>
NSL::Lattice::Square<Type>::Square(
// scalar kappa and spacing
            const std::vector<std::size_t> n,
            const Type & kappa,
            const double spacing
            ):
        NSL::Lattice::SpatialLattice<Type>(
                name(n),
                NSL::Tensor<Type>(this->n_to_sites_(n), this->n_to_sites_(n)),
                NSL::Tensor<double>(this->n_to_sites_(n), n.size())
        ),
        dimensions_(n),
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
NSL::Lattice::Square<Type>::Square(
// scalar kappa
            const std::vector<std::size_t> n,
            const Type & kappa,
            const std::vector<double> spacings
            ):
        NSL::Lattice::SpatialLattice<Type>(
                name(n),
                NSL::Tensor<Type>(this->n_to_sites_(n), this->n_to_sites_(n)),
                NSL::Tensor<double>(this->n_to_sites_(n), n.size())
        ),
        dimensions_(n),
        integers_(integer_coordinates_(n))
{
    std::vector<Type> kappas(n.size());
    for(std::size_t i=0; i < kappas.size(); ++i){
        kappas[i] = kappa;
    }

    this->init_(n, kappas, spacings);
}

} // namespace NSL::Lattice
#endif //NSL_LATTICE_SQUARE_HPP
