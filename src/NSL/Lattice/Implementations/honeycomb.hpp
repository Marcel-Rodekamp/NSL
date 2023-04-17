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
                const std::vector<std::size_t L,
                const Type &kappas = 1.);

    protected:
        //! Coordinate space lattice vectors
        NSL::Tensor<double> a = {
            {1.5, +0.8660254037844386}, // √(3) / 2 {√3, +1}
            {1.5, -0.8660254037844386}  // √(3) / 2 {√3, -1}
        };
        // Separation between sites in the unit cell.
        NSL::Tensor<double> r = {
            1., 0.
        };
        //! Reciprocal space lattice vectors
        NSL::Tensor<double> b = {
            {2.094395102393195, 3.627598728468436}, // 2π/√3 {1/√3, +1}
            {2.094395102393195, -3.627598728468436} // 2π/√3 {1/√3, -1}
        };

        std::vector<std::size_t> L;
        std::size_t unit_cells;

        //! Coordinates
        NSL::Tensor<Type> coordinates_;
        Type kappa;

    private:
        static inline double distance_squared(const NSL::Tensor<double> &x, const NSL::Tensor<double> &y);
        static inline std::size_t unit_cells_(const std::vector<std::size_t> &L);
        static inline std::size_t n_to_sites_(const std::vector<std::size_t> &n);
        static NSL::Tensor<int> integer_coordinates_(const std::vector<std::size_t> &n);
        void init_(const std::vector<std::size_t> &n,
                   const std::vector<Type> &kappa,
                   const std::vector<double> spacings);
};

#endif // NSL_LATTICE_HONEYCOMB_HPP
