#ifndef NSL_LATTICE_COMPLETE_HPP
#define NSL_LATTICE_COMPLETE_HPP

#include "../lattice.hpp"

namespace NSL::Lattice {

/*! The complete graph of n sites.
 *  For convenience, geometrically distributed around a circle.
 **/
template <typename Type>
class Complete: public NSL::Lattice::SpatialLattice<Type> {
    public:
        /*!
         *  \param n the number of sites
         *  \param kappa the hopping amplitude.
         *  \param radius 
         **/
        explicit Complete(const std::size_t n, const Type & kappa = 1, const double &radius = 1);

        bool bipartite() { return !(this->sites() > 2); }
};

template <typename Type>
class Triangle: public NSL::Lattice::Complete<Type> {
    public:
        explicit Triangle(  const Type & kappa = 1,
                            const double radius = 1):
            NSL::Lattice::Complete<Type>(3, kappa, radius) {
            // this->name_ = "Triangle";
            };
        bool bipartite() { return false; };
};

template <typename Type>
class Tetrahedron: public NSL::Lattice::Complete<Type> {
    public:
        explicit Tetrahedron(  const Type & kappa = 1,
                               const double edge  = 1):
            NSL::Lattice::Complete<Type>(4, kappa) {
                // this->name_ = "Tetrahedron";
                // With a unit side length, the tetrahedron has coordinates
                // given by any O(3) rotation of the following
                //
                // [0,0, sqrt(2/3) - 1 / 2 sqrt(6) ]
                this->sites_(0,0) = 0.;
                this->sites_(0,1) = 0.;
                this->sites_(0,2) = 0.6123724356957945;
                // [-1/2 sqrt(3), -1/2, - 1 / 2 sqrt(6) ]
                this->sites_(1,0) = -0.2886751345948129;
                this->sites_(1,1) = -0.5;
                this->sites_(1,2) = -0.2041241452319315;
                // [-1/2 sqrt(3), +1/2, - 1 / 2 sqrt(6) ]
                this->sites_(2,0) = -0.2886751345948129;
                this->sites_(2,1) = +0.5;
                this->sites_(2,2) = -0.2041241452319315;
                // [1/sqrt(3), 0, - 1 / 2 sqrt(6) ]
                this->sites_(3,0) = 0.5773502691896258;
                this->sites_(3,1) = 0.;
                this->sites_(3,2) = -0.2041241452319315;
                // These were generated using Mathematica,
                // PolyhedronCoordinates[Tetrahedron[]]

                // For an arbitrary side length, just scale.
                this->sites_ *= edge;
            };

        bool bipartite() { return false; };
};


} // namespace NSL
#endif
