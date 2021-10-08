#ifndef NSL_LATTICE_SQUARE_HPP
#define NSL_LATTICE_SQUARE_HPP

#include "../lattice.hpp"

namespace NSL::Lattice {

/*! Sites evenly spaced on orthogonal coorinates, starting at 0.
 **/
template <typename Type>
class Square: public NSL::Lattice::SpatialLattice<Type> {
    public:
        /*!
         *  \param n the number of sites in each direction
         *  \param spacing
         **/
        explicit Square(
                const std::vector<std::size_t> n,
                const Type & kappa = 1,
                const double spacing = 1);
        // explicit Square(
        //         const std::vector<std::size_t> n,
        //         const Type & kappa,
        //         const std::vector<double> spacing);
        // explicit Square(const std::vector<std::size_t> n, NSL::Lattice::Boundary condition);
        // explicit Square(const std::vector<std::size_t> n, const std::vector<double> spacings);
        // explicit Square(const std::vector<std::size_t> n, const std::vector<double> spacings, const std::vector<NSL::Lattice::Boundary> condition);

    protected:
        NSL::Tensor<int> integers_;
    private:
        static inline std::size_t n_to_sites_(const std::vector<std::size_t> &n);
        static NSL::Tensor<int> integer_coordinates_(const std::vector<std::size_t> &n);
        void init_(const std::vector<std::size_t> &n,
                   const std::vector<Type> &kappa,
                   const std::vector<double> spacings);
};

} // namespace NSL
#endif
