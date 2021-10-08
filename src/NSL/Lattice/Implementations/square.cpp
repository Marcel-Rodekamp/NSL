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
inline std::size_t NSL::Lattice::Square<Type>::n_to_sites(const std::vector<std::size_t> &n){
    std::size_t volume=1;
    for(const auto& value: n) volume *= value;
    return volume;
}

template<typename Type>
NSL::Tensor<int> NSL::Lattice::Square<Type>::integer_coordinates(const std::vector<std::size_t> &n){
    std::size_t sites = NSL::Lattice::Square<Type>::n_to_sites(n);
    std::size_t dimensions = n.size();
    NSL::Tensor<int> coordinates(sites, dimensions);

    std::size_t copies = 1;
    std::size_t run = sites;
    for(std::size_t d = 0; d < dimensions; ++d) {
        run /= n[d];
        for(std::size_t copy = 0; copy < copies; ++copy){
            for(std::size_t x = 0; x < n[d]; ++x) {
                for(std::size_t repeat=0; repeat < run; ++repeat){
                    coordinates(copy*run*n[d]+x*run + repeat, d) = x;
                }
            }
        }
        copies *= n[d];
    }
    return coordinates;
}

template<typename Type>
NSL::Lattice::Square<Type>::Square(const std::vector<std::size_t> n):
        NSL::Lattice::SpatialLattice<Type>(
                "Square()",    //! todo: stringify
                NSL::Tensor<Type>(64, 64),
                NSL::Tensor<double>(64, 4)
                //NSL::Tensor<Type>(this->n_to_sites(n), this->n_to_sites(n)),
                //NSL::Tensor<double>(this->n_to_sites(n), n.size())
        )
{
    auto sites = this->n_to_sites(n);
    NSL::Tensor<int> x = this->integer_coordinates(n);
    for(int i = 0; i < sites; ++i) {
        for(int d = 0; d < n.size(); ++d) {
            this->sites_(i,d) = x(i,d);
        }
    }

    for (int i = 0; i < sites; ++i) {
        this->hops_(i,i) = 0;
        // this->hops_(i, i + 1) = kappa;
    }

}

} // namespace NSL::Lattice

template class NSL::Lattice::Square<float>;
template class NSL::Lattice::Square<double>;

#endif
