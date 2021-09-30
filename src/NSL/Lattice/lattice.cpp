#include "lattice.hpp"
#include "../LinAlg/mat_exp.hpp"

namespace NSL::Lattice {

template <typename Type>
NSL::Lattice::SpatialLattice<Type>::SpatialLattice(
    const std::string &name,
    const NSL::Tensor<Type> &hops,
    const NSL::Tensor<double> &sites
    ) :
    name_(name),
    hops_(hops),
    sites_(sites)
    {
        // TODO: assert that hops_ is a square matrix, size matches sites_.
    }

template <typename Type>
const NSL::Tensor<double> NSL::Lattice::SpatialLattice<Type>::operator()(size_t index){
    assertm( (0 <= index) && (index < this->sites()),
            "Spatial lattice site out of bounds.");
    return this->sites_.slice(0,index,index+1,1);
}

//size_t SpatialLattice::operator()(const & NSL::Site x){
//  TODO
//}

template <typename Type>
size_t NSL::Lattice::SpatialLattice<Type>::sites(){ return sites_.shape(0); }

template <typename Type>
NSL::Tensor<Type> NSL::Lattice::SpatialLattice<Type>::adjacency_matrix(){
    if (!(this->adj_is_initialized_)) this->compute_adjacency(this->hops_);
    return adj_;
}

template <typename Type>
NSL::Tensor<Type> NSL::Lattice::SpatialLattice<Type>::hopping_matrix(Type delta){
    return delta == 1. ? hops_ : hops_ * delta;
}

template <typename Type>
NSL::Tensor<Type> NSL::Lattice::SpatialLattice<Type>::exp_hopping_matrix(Type delta){
    if(! exp_hopping_matrix_.contains(delta)){
        // compute if it's not in exp_hopping_matrix_ already
        this->exp_hopping_matrix_[delta] = NSL::LinAlg::mat_exp(this->hopping_matrix(delta));
    }
    return this->exp_hopping_matrix_[delta];
}

template <typename Type>
void NSL::Lattice::SpatialLattice<Type>::compute_adjacency(const NSL::Tensor<Type> & hops) {
    // TODO: actually compute the adjacency: 1 if connected, 0 if not.
    this->adj_ = hops;
}

} // namespace NSL

template class NSL::Lattice::SpatialLattice<float>;
template class NSL::Lattice::SpatialLattice<double>;
