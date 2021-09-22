#ifndef NSL_LATTICE_HPP
#define NSL_LATTICE_HPP

#include "../assert.hpp"
#include "../complex.hpp"

#include "../Tensor/tensor.hpp"

namespace NSL {
namespace Lattice {

struct Site{
}; // Site

template <typename Type>
class SpatialLatticeBase {
    public:
        SpatialLatticeBase(const NSL::Tensor<Type> & hops, const std::vector<NSL::Lattice::Site> sites);

        NSL::Lattice::Site & operator()(size_t index);
        // size_t operator()(const & NSL::Site x);

        size_t sites();

        NSL::Tensor<Type> adjacency_matrix();
        NSL::Tensor<Type> hopping_matrix(Type delta=1.);
        NSL::Tensor<Type> exp_hopping_matrix(Type delta=1.);

        const std::string & name() { return name_; };

    private:
        const NSL::Tensor<Type> adj_;
        const NSL::Tensor<Type> hops_;
        const std::vector<NSL::Lattice::Site> sites_;
        std::map<double,NSL::Tensor<Type>> exp_hopping_matrix_;
        const std::string name_;
        
        static NSL::Tensor<Type> & compute_adjacency(const NSL::Tensor<Type> & hops);

    }; // SpatialLatticeBase

class Lattice {
    public:

    private:
    }; // Lattice

} // namespace Lattice
} // namespace NSL

#endif
