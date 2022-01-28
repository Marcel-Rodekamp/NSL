#ifndef NSL_LATTICE_HPP
#define NSL_LATTICE_HPP

/*! \file lattice.hpp
 *  Classes for spatial and spacetime lattices.
 *
 *  We separate the spacetime construction in our computation into
 *  spacetimes (which are stacks of the same spatial lattice),
 *  spatial lattices (which are spatial sites connected together),
 *  and
 *  sites.
 **/

#include "../assert.hpp"
#include "../complex.hpp"
#include "../map.hpp"
#include "../Tensor/tensor.hpp"

namespace NSL::Lattice {

/*! A base class for spatial lattices (ie. finite graphs of Sites).
 *      Offers a variety of default methods, which might be overridden by
 *      children classes (if, for example, there is a more direct / stable 
 *      way to compute some result).
 *
 *      For example, gives generic methods for getting the adjacency matrix
 *      from the hopping matrix, computing and memoizing exp.
 **/
template <typename Type>
class SpatialLattice {
    public:
        //  No default constructor
        /*  There is no default constructor.
         *  You must talk about SOME particular lattice. */
        SpatialLattice() = delete;

        //! A minimal constructor.
        /*! 
         *  \param name  a string which describes this lattice.
         *  \param hops  a sparse or dense matrix of hopping amplitudes.
         *  \param sites a vector of sites; the indices match those in `hops`.
         **/
        SpatialLattice(const std::string & name, const NSL::Tensor<Type> & hops, const NSL::Tensor<double> & sites);

        /*!
         *  \param index is a linear index into the vector of sites in the lattice.
         *  \returns a Site in the lattice.
         **/
        const NSL::Tensor<double> operator()(size_t index);

        //! Gives a list of coordinates in the order of sites.
        const NSL::Tensor<double> coordinates(){
            return this->sites_;
        };

        // size_t operator()(const NSL::Tensor<double> &x);

        //! Gives the number of spatial sites.
        size_t sites();

        //! Give the adjacency matrix of the lattice, 1 if the sites are connected, 0 otherwise.
        /*!
         *  \todo change the type signature to `NSL::Tensor<int>` or `<bool>`.
         *  `<int>` probably makes sense, because you might need to perform arithmetic
         *  operations to deduce graph-theoretic properties.  For example
         *  diagonal(adjacency^(all odd powers)) = 0 if the graph is bipartite.
         **/
        NSL::Tensor<int> adjacency_matrix();

        //! Give the matrix of hopping amplitudes. Can be complex, as long as it's Hermitian.
        /*!
         *  \param delta what factor to multiply by;
         *               typically the Trotter discretization.
         **/
        NSL::Tensor<Type> hopping_matrix(Type delta=1.);

        //! Give the matrix exponential hopping amplitudes.
        /*!
         *  \param delta what factor to multiply by before exponentiating;
         *               typically the Trotter discretization.
         **/
        NSL::Tensor<Type> exp_hopping_matrix(Type delta=1.);

        //! A string that describes the lattice.
        const std::string & name() { return name_; };

        bool bipartite();

    protected:
        //! A descriptive string for quick human identification of the lattice.
        const std::string name_;
        //! The hopping matrix.
        NSL::Tensor<Type> hops_;
        //! The sites connectd by the hopping amplitudes.
        NSL::Tensor<double> sites_;
        //! Since exponentiating can be costly, a place to memoize results.
        NSL::map<Type,NSL::Tensor<Type>> exp_hopping_matrix_;
        //! Store whether the lattice can be bipartitioned.
        bool bipartite_ = false;
        //! We only check for bipartiteness on the first request.
        bool bipartite_is_initialized_ = false;
        // Maybe the right thing is to have wrapper that can hold
        // a value or be Uninitialized

    private:
        //! We only compute the adjacency matrix on first request.
        bool adj_is_initialized_ = false;
        //! The (computed) adjacency matrix.
        NSL::Tensor<int> adj_;
        //! Transform the hopping matrix into the adjacency matrix.
        /*!
         *  \param hops a matrix of hopping amplitudes.
         *         Zero amplitudes imply non-adjacent sites,
         *         nonzero amplitudes imply adjacency.
         **/
        void compute_adjacency(NSL::Tensor<Type> hops);
        
        // A generic method, in case no short-cut is available
        void compute_bipartite();
    }; // SpatialLattice

class Lattice {
    public:

    private:
    }; // Lattice

} // namespace NSL

#endif
