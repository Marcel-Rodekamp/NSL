#ifndef NSL_LATTICE_HPP
#define NSL_LATTICE_HPP

#include "../assert.hpp"
#include "../complex.hpp"

#include "../Tensor/tensor.hpp"

namespace NSL::Lattice {

struct Site{
}; // Site

template <typename Type>
class SpatialLatticeBase {
    public:
        SpatialLatticeBase() = delete;
        SpatialLatticeBase(const std::string & name, const NSL::Tensor<Type> & hops, const std::vector<NSL::Lattice::Site> & sites);


        const NSL::Lattice::Site operator()(size_t index);
        // size_t operator()(const & NSL::Site x);

        size_t sites();

        NSL::Tensor<Type> adjacency_matrix();
        NSL::Tensor<Type> hopping_matrix(Type delta=1.);
        NSL::Tensor<Type> exp_hopping_matrix(Type delta=1.);

        const std::string & name() { return name_; };

    protected:
        const std::string name_;
        NSL::Tensor<Type> adj_;
        NSL::Tensor<Type> hops_;
        std::vector<NSL::Lattice::Site> sites_;
        std::map<double,NSL::Tensor<Type>> exp_hopping_matrix_;

    private:
        bool adj_is_initialized_ = false;

        void compute_adjacency(const NSL::Tensor<Type> & hops);
    }; // SpatialLatticeBase

class Lattice {
    public:

    private:
    }; // Lattice

} // namespace NSL

#endif
