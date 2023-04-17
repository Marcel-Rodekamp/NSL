#ifndef NSL_LATTICE_HONEYCOMB_TPP
#define NSL_LATTICE_HONEYCOMB_TPP


/*! \file honeycomb.tpp
 *  \author Evan Berkowitz
 *  \date April 2023
 *  \brief Implementation of the honeycomb lattice for graphene.
*/

#include <cmath>

#include "honeycomb.hpp"

// We describe the honeycomb lattice by a pair of integers L1, and L2,
namespace {
    std::string name(const std::vector<std::size_t> &L){
        std::string result="Honeycomb(" + std::to_string(L[0]) +", " + std::to_string(L[1]) + ")";
        return result;
    }
}
// which we group into
// a single std::vector<std::size_t> of length 2.  L1 and L2 multiply the lattice vectors
//
//  a1 = √(3) / 2 * [√3, +1]
//  a2 = √(3) / 2 * [√3, -1]
//
// which has reciprocal vectors
//
// b1 = 2π/√3 * [1/√3, +1]
// b2 = 2π/√3 * [1/√3, -1]
//
// so that ai.bj = 2π δ(i,j)
namespace NSL::Lattice {

// There are L1 * L2 unit cells,
template<typename Type>
inline std::size_t NSL::Lattice::Honeycomb<Type>::unit_cells_(const std::vector<std::size_t> &L){
    std::size_t volume=1;
    for(const auto& value: L) volume *= value;
    return volume;
}
// and two sites per unit cell.
template<typename Type>
inline std::size_t NSL::Lattice::Honeycomb<Type>::L_to_sites_(const std::vector<std::size_t> &L){
    return 2*unit_cells_(L);
}

// Because we use periodic boundary conditions, the distance calculation is nontrivial,
template<typename Type>
double NSL::Lattice::Honeycomb<Type>::distance_squared(const NSL::Tensor<double> &x, const NSL::Tensor<double> &y){

    NSL::Tensor<double> L1 = self->L[0] * self->a[0];
    NSL::Tensor<double> L2 = self->L[0] * self->a[0];

    // The farthest things can be (with periodic boundary conditions) is much less than
    NSL::Tensor<double> vector = L1+L2;
    double distance = NSL::LinAlg::inner_product(vector, vector);
    double min = distance;

    // Now we check all possibilities of having to go across the boundary.
    // 'sign' is maybe a bit funny for a name, it can be 0 too, if you don't
    // cross the boundary.
    for(int sign1 = -1; sign1 <= 1; sign1++){
        for(int sign2 = -1; sign2 <= 1; sign2++){
            vector = x-y+sign1*L1 + sign2*L2;
            distance = NSL::LinLAlg::inner_product(vector, vector);
            min = (min < distance) ? min : distance;
        }
    }
    return min;
}


template<typename Type>
void NSL::Lattice::Honeycomb<Type>::init_(const std::vector<std::size_t> &L,
                   const Type &kappa)
{

    this->kappa = kappa;
    this->spacings_ = spacings;

    for(int L1 = 0; L1 < L[0]; ++L1){
        for(int L2 = 0; L2 < L[1]; ++L2){
            i = L1 + L[0]*L2;
            // Now alternate over the A/B structure
            this->sites_(2*i+0) = NSL::LinAlg::mat_mul(this->a, L) + this->r/2;
            this->sites_(2*i+1) = NSL::LinAlg::mat_mul(this->a, L) - this->r/2;
        }
    }

    //! todo this algorithm is quadratic in the number of sites.
    // I think, with intelligence, it can be made linear.
    // But, it works for now.
    for (int i = 0; i < this->sites(); ++i){
        for (int j = i; j < this->sites(); ++j){
            double distance_squared = this->distance_squared(sites_(i) - sites_(j))

            // Typically for 0ness we would check a small cutoff like
            // 1e-12.  But we already know, by the geometry and correctness
            // of the coordinates, that this is 'close enough'.
            if( (distance_squared-1).abs() < 0.01){
                this->hops_(i,j) = kappa;
                this->hops_(j,i) = kappa;
                }
        }
    }

    this->compute_adjacency();

}

//=====================================================================
// Constructors
//=====================================================================

template<typename Type>
NSL::Lattice::Honeycomb<Type>::Honeycomb(
            const std::vector<std::size_t> L,
            const Type & kappa
            ):
        NSL::Lattice::SpatialLattice<Type>(
                name(L),
                NSL::Tensor<Type>(this->n_to_sites_(n), this->n_to_sites_(n)),
                NSL::Tensor<double>(this->n_to_sites_(n), 2)
        ),
        L(L),
        unit_cells(this->unit_cells_(L))
{
    this->init_(L, kappa);
}

} // namespace NSL::Lattice
#endif //NSL_LATTICE_HONEYCOMB_TPP
