#ifndef NSL_LATTICE_HONEYCOMB_HPP
#define NSL_LATTICE_HONEYCOMB_HPP

#include "../lattice.hpp"

namespace NSL::Lattice {

/*! The honeycomb lattice of graphene.
 **/
template <typename Type>
class Honeycomb: public NSL::Lattice::SpatialLattice<Type> {
    public:
        /*!
         *  \param L the number of sites in the a1 and a2 directions
         *  \param kappas the hopping parameter in each direction
         *
         **/
        explicit Honeycomb(
                const std::vector<int> L,
                const Type &kappa = 1.);

        const NSL::Tensor<int> L = NSL::Tensor<int>(2);
        const int unit_cells;

    protected:
        //! Coordinate space lattice vectors
        //  √(3) / 2 {
        //      {√3, +1}
        //      {√3, -1}
        //      }
        NSL::Tensor<double> a = NSL::Tensor<double>(2,2);
        
        // Separation between sites in the unit cell.
        // {1., 0.}
        NSL::Tensor<double> r = NSL::Tensor<double>(2);
        
        //! Reciprocal space lattice vectors
        //  2π/√3 {
        //      {1/√3, +1}
        //      {1/√3, -1}
        //      }
        NSL::Tensor<double> b = NSL::Tensor<double>(2,2);

        Type kappa;

    private:
        double distance_squared(const NSL::Tensor<double> &x, const NSL::Tensor<double> &y);
        inline int unit_cells_(const std::vector<int> &L);
        inline int L_to_sites_(const std::vector<int> &L);
        void init_();
};

} // namespace NSL::Lattice

#endif // NSL_LATTICE_HONEYCOMB_HPP
