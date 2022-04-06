#ifndef NSL_LATTICE_RING_HPP
#define NSL_LATTICE_RING_HPP

#include "../lattice.hpp"

namespace NSL::Lattice {

/*! A platonic solid with 20 vertices, 30 edges, and 12 pentagonal faces.  Nearest-neighbor hopping on the edges.
 *
 * With a little bit of drawing, you can convince yourself that the topology of the nearest-neighbor connections cannot be complex if the hopping matrix is to be hermitian.
 * Therefore this class can only take a real type.
 **/

//@TODO Restrict this to real types only.  May require an isReal concept.
template <typename Type>
class Dodecahedron: public NSL::Lattice::SpatialLattice<Type> {
    public:
        /*!
         *  \param kappa the hopping amplitude.
         *  \param side_length
         **/
        explicit Dodecahedron(const Type & kappa = 1, const double &side_length= 1);
    private:
        const double side_length_
};



template<typename Type>
NSL::Lattice::Dodecahedron<Type>::Ring(const Type &kappa, const double &side_length) :
        NSL::Lattice::SpatialLattice<Type>(
                "Dodecahedron",
                NSL::Tensor<Type>(20, 20),
                NSL::Tensor<double>(20,3)
        ),
        side_length_(side_length)
{

    // Generated in Mathematica via 
    //     sites = N[#, 16] & /@ FullSimplify[PolyhedronData["Dodecahedron", "VertexCoordinates"]]
    // Has minimum of 1 between sites,
    //     Outer[N@Norm[ExpandAll[#1 - #2]] &, sites, sites, 1]
    // (except for the diagonal which is trivially 0).
    this->sites_( 0,0) = -1.376381920471174;  this->sites_( 0,1) = 0.;                  this->sites_( 0,2) = +0.2628655560595668;
    this->sites_( 1,0) = +1.376381920471174;  this->sites_( 1,1) = 0.;                  this->sites_( 1,2) = -0.2628655560595668;
    this->sites_( 2,0) = -0.4253254041760200; this->sites_( 2,1) = -1.309016994374947;  this->sites_( 2,2) = +0.2628655560595668;
    this->sites_( 3,0) = -0.4253254041760200; this->sites_( 3,1) = +1.309016994374947;  this->sites_( 3,2) = +0.2628655560595668;
    this->sites_( 4,0) = +1.113516364411607;  this->sites_( 4,1) = -0.8090169943749474; this->sites_( 4,2) = +0.2628655560595668;
    this->sites_( 5,0) = +1.113516364411607;  this->sites_( 5,1) = +0.8090169943749474; this->sites_( 5,2) = +0.2628655560595668;
    this->sites_( 6,0) = -0.2628655560595668; this->sites_( 6,1) = -0.8090169943749474; this->sites_( 6,2) = +1.113516364411607;
    this->sites_( 7,0) = -0.2628655560595668; this->sites_( 7,1) = +0.8090169943749474; this->sites_( 7,2) = +1.113516364411607;
    this->sites_( 8,0) = -0.6881909602355868; this->sites_( 8,1) = -0.5;                this->sites_( 8,2) = -1.113516364411607;
    this->sites_( 9,0) = -0.6881909602355868; this->sites_( 9,1) = +0.5;                this->sites_( 9,2) = -1.113516364411607;
    this->sites_(10,0) = +0.6881909602355868; this->sites_(10,1) = -0.5;                this->sites_(10,2) = +1.113516364411607;
    this->sites_(11,0) = +0.6881909602355868; this->sites_(11,1) = +0.5;                this->sites_(11,2) = +1.113516364411607;
    this->sites_(12,0) = +0.8506508083520399; this->sites_(12,1) = 0.;                  this->sites_(12,2) = -1.113516364411607;
    this->sites_(13,0) = -1.113516364411607;  this->sites_(13,1) = -0.8090169943749474; this->sites_(13,2) = -0.2628655560595668;
    this->sites_(14,0) = -1.113516364411607;  this->sites_(14,1) = +0.8090169943749474; this->sites_(14,2) = -0.2628655560595668;
    this->sites_(15,0) = -0.8506508083520399; this->sites_(15,1) = 0.;                  this->sites_(15,2) = +1.113516364411607;
    this->sites_(16,0) = +0.2628655560595668; this->sites_(16,1) = -0.8090169943749474; this->sites_(16,2) = -1.113516364411607;
    this->sites_(17,0) = +0.2628655560595668; this->sites_(17,1) = +0.8090169943749474; this->sites_(17,2) = -1.113516364411607;
    this->sites_(18,0) = +0.4253254041760200; this->sites_(18,1) = -1.309016994374947;  this->sites_(18,2) = -0.2628655560595668;
    this->sites_(19,0) = +0.4253254041760200; this->sites_(19,1) = +1.309016994374947;  this->sites_(19,2) = -0.2628655560595668;

    this->sites_*=side_length;

    // The adjacency is given by 
    //     PolyhedronData["Dodecahedron", "EdgeIndices"] - 1
    // (the -1 is because Mathematica uses 1-indexing).
    this->hops_( 0,13) = kappa;
    this->hops_( 0,14) = kappa;
    this->hops_( 0,15) = kappa;
    this->hops_( 1, 4) = kappa;
    this->hops_( 1, 5) = kappa;
    this->hops_( 1,12) = kappa;
    this->hops_( 2, 6) = kappa;
    this->hops_( 2,13) = kappa;
    this->hops_( 2,18) = kappa;
    this->hops_( 3, 7) = kappa;
    this->hops_( 3,14) = kappa;
    this->hops_( 3,19) = kappa;
    this->hops_( 4,10) = kappa;
    this->hops_( 4,18) = kappa;
    this->hops_( 5,11) = kappa;
    this->hops_( 5,19) = kappa;
    this->hops_( 6,10) = kappa;
    this->hops_( 6,15) = kappa;
    this->hops_( 7,11) = kappa;
    this->hops_( 7,15) = kappa;
    this->hops_( 8, 9) = kappa;
    this->hops_( 8,13) = kappa;
    this->hops_( 8,16) = kappa;
    this->hops_( 9,14) = kappa;
    this->hops_( 9,17) = kappa;
    this->hops_(10,11) = kappa;
    this->hops_(12,16) = kappa;
    this->hops_(12,17) = kappa;
    this->hops_(16,18) = kappa;
    this->hops_(17,19) = kappa;

    // Since kappa must be real anyway we can simply add the transpose.
    this->hops_ += this->hops_.transpose();

    this->compute_adjacency();
}

// TODO: Evan knows the rotational symmetries, irreps, the states that may mix, etc.

} // namespace NSL::Lattice

#endif
