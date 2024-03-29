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
         *  \param kappas the hopping parameters in each direction
         *  \param spacings the lattice spacings in each direction
         *
         *  All three arguments must have the same length.
         **/
        explicit Square(
                const std::vector<std::size_t> n,
                const std::vector<Type> &kappas,
                const std::vector<double> spacings);
        /*!
         *  \param n the number of sites in each direction
         *  \param kappas the hopping parameters in each direction
         *  \param spacing a common lattice spacing
         *
         *  `n` and `kappas` must have the same length.
         **/
        explicit Square(
                const std::vector<std::size_t> n,
                const std::vector<Type> &kappas,
                const double spacing = 1);
        /*!
         *  \param n the number of sites in each direction
         *  \param kappa a common hopping parameter
         *  \param spacing a common lattice spacing
         **/
        explicit Square(
                const std::vector<std::size_t> n,
                const Type & kappa = 1,
                const double spacing = 1);
        /*!
         *  \param n the number of sites in each direction
         *  \param kappa a common hopping parameter
         *  \param spacings the lattice spacings in each direction.
         *
         *  `n` and `spacings` must have the same length.
         **/
        explicit Square(
                const std::vector<std::size_t> n,
                const Type & kappa,
                const std::vector<double> spacings);

    protected:
        //! How many sites in each orthogonal direction.
        std::vector<std::size_t> dimensions_;
        //! Integer coordinates, from (0,0,0...) to (dimensions), left-most slowest.
        NSL::Tensor<int> integers_;
        std::vector<Type> kappas_;
        std::vector<double> spacings_;
    private:
        static inline std::size_t n_to_sites_(const std::vector<std::size_t> &n);
        static NSL::Tensor<int> integer_coordinates_(const std::vector<std::size_t> &n);
        void init_(const std::vector<std::size_t> &n,
                   const std::vector<Type> &kappa,
                   const std::vector<double> spacings);
};

/*! Sites evenly spaced in 3D on orthogonal coorinates,
 *  starting at (0,0,0), and filling the first octant,
 *  with equal hopping amplitude and lattice spacing in each direction.
 **/
template <typename Type>
class Cube3D: public NSL::Lattice::Square<Type> {
    public:
        /*!
         *  \param n the number of sites in each direction
         *  \param kappa a common hopping parameter
         *  \param spacing a common lattice spacing
         **/
        explicit Cube3D(
                    std::size_t n,
                    const Type & kappa = 1,
                    const double spacing = 1): 
                NSL::Lattice::Square<Type>(
                        std::vector<std::size_t>({n,n,n}),
                        kappa,
                        spacing)
                {};
};

} // namespace NSL
#endif
