#ifndef NSL_LATTICE_HONEYCOMB_TPP
#define NSL_LATTICE_HONEYCOMB_TPP


/*! \file honeycomb.tpp
 *  \author Evan Berkowitz
 *  \date   April 2023
 *  \brief  Implementation of the honeycomb lattice for graphene.
*/

#include <cmath>

#include "honeycomb.hpp"
#include "LinAlg/inner_product.tpp"

// We describe the honeycomb lattice by a pair of integers L1, and L2,
// which we group into a single std::vector<int>.
namespace {
    std::string nameHoneycomb(const std::vector<int> &L){
        std::string result="Honeycomb(" + std::to_string(L[0]) +", " + std::to_string(L[1]) + ")";
        return result;
    }

    NSL::Tensor<int> honeycomb_L_converter(const std::vector<int> &L){
        NSL::Tensor<int> l(2);
        l(0) = L[0];
        l(1) = L[1];
        return l;
    }
}

namespace NSL::Lattice {

// There are L1 * L2 unit cells,
template<typename Type>
inline int NSL::Lattice::Honeycomb<Type>::unit_cells_(const std::vector<int> &L){
    int volume=L[0] * L[1];
    return volume;
}
// and two sites per unit cell.
template<typename Type>
inline int NSL::Lattice::Honeycomb<Type>::L_to_sites_(const std::vector<int> &L){
    return 2*unit_cells_(L);
}

// Because we use periodic boundary conditions, the distance calculation is nontrivial,
template<typename Type>
double Honeycomb<Type>::distance_squared(const NSL::Tensor<double> &x, const NSL::Tensor<double> &y){

    NSL::Tensor<double> L1 = this->L(0) * this->a(0, NSL::Slice());
    NSL::Tensor<double> L2 = this->L(1) * this->a(1, NSL::Slice());

    // The farthest things can be (with periodic boundary conditions) is much less than
    NSL::Tensor<double> vector = L1+L2;
    double distance = NSL::LinAlg::inner_product(vector, vector);
    double min = distance;

    // Now we check all possibilities of having to go across the boundary.
    // 'sign' is maybe a bit funny for a name, it can be 0 too, if you don't
    // cross the boundary.
    for(int sign1 = -1; sign1 <= 1; sign1++){
        for(int sign2 = -1; sign2 <= 1; sign2++){
            vector   = x - y + sign1*L1 + sign2*L2;
            distance = NSL::LinAlg::inner_product(vector, vector);
            min      = (min < distance) ? min : distance;
        }
    }
    return min;
}


template<typename Type>
void NSL::Lattice::Honeycomb<Type>::init_()
{
    this->a(0,0) = +1.5;                    // +3/2
    this->a(0,1) = +0.8660254037844386;     // +√(3)/2
    this->a(1,0) = +1.5;                    // +3/2
    this->a(1,1) = -0.8660254037844386;     // -√(3)/2

    this->r(0) = 1.;
    this->r(1) = 0.;
    
    this->b(0,0) = +2.094395102393195;      // +2π/3
    this->b(0,1) = +3.627598728468436;      // +2π/√3
    this->b(1,0) = +2.094395102393195;      // +2π/3
    this->b(1,1) = -3.627598728468436;      // -2π/√3

    for(int L1 = 0; L1 < this->L(0); ++L1){
        for(int L2 = 0; L2 < this->L(1); ++L2){
            int i = L1 + this->L(0)*L2;
            // Now alternate over the A/B structure
            this->sites_(2*i+0, NSL::Slice()) = L1 * a(0, NSL::Slice()) + L2 * a(1, NSL::Slice()) + this->r/2;
            this->sites_(2*i+1, NSL::Slice()) = L1 * a(0, NSL::Slice()) + L2 * a(1, NSL::Slice()) - this->r/2;
        }
    }

    //! todo this algorithm is quadratic in the number of sites.
    // I think, with intelligence, it can be made linear.
    // But, it works for now.
    for (int i = 0; i < this->sites(); ++i){
        for (int j = i; j < this->sites(); ++j){
            double distance_squared = this->distance_squared(this->sites_(i, NSL::Slice()), this->sites_(j, NSL::Slice()));

            // Typically for 0ness we would check a small cutoff like
            // 1e-12.  But we already know, by the geometry and correctness
            // of the coordinates, that this is 'close enough'.
            if( NSL::LinAlg::abs(distance_squared-1) < 0.0001){
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
            const std::vector<int> L,
            const Type & kappa
            ):
        NSL::Lattice::SpatialLattice<Type>(
                nameHoneycomb(L),
                NSL::Tensor<Type>(this->L_to_sites_(L), this->L_to_sites_(L)),
                NSL::Tensor<double>(this->L_to_sites_(L), 2)
        ),
        unit_cells(this->unit_cells_(L)),
        L(honeycomb_L_converter(L)), //! todo This is a dirty hack some day we should have constructor std::vector -> NSL::Tensor 
        kappa(kappa)
{

    this->init_();
}

} // namespace NSL::Lattice
#endif //NSL_LATTICE_HONEYCOMB_TPP
