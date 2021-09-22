namespace NSL {
namespace Lattice {

template <typename Type>
NSL::SpatialLatticeBase(
        const & std::string name,
        const & NSL::Tensor<Type> hops,
        const std::vector<NSL::Lattice::Site> & sites):
    name_(name),
    hops_(hops),
    sites_(sites),
    adj_(compute_adjacency(hops))
    {
        // TODO: assert that hops_ is a square matrix, size matches sites_.
}

NSL::Lattice::Site & SpatialLatticeBase::operator()(size_t index){
    assert( (0 <= index) && (index < volume_),
            "Spatial lattice site out of bounds.");
    return sites_[index];
}

//size_t SpatialLatticeBase::operator()(const & NSL::Site x){
//  TODO
//}

size_t SpatialLatticeBase::sites(){ return sites_.size(); }

template <typename Type>
NSL::Tensor<Type> SpatialLatticeBase::adjacency_matrix(){
    return adj_;
}

template <typename Type>
NSL::Tensor<Type> SpatialLatticeBase::hopping_matrix(Type delta=1.){
    return delta == 1. ? hops_ : delta * hops_;
}

template <typename Type>
NSL::Tensor<Type> SpatialLatticeBase::exp_hopping_matrix(Type delta=1.){
    if(! exp_hopping_matrix_.contains(delta)){
        // compute if it's not in exp_hopping_matrix_ already
        exp_hopping_matrix_[delta] = NSL::LinAlg::mat_exp(hopping_matrix(delta));
    }
    return exp_hopping_matrix[delta];
}

template <typename Type>
static NSL::Tensor<Type> & compute_adjacency(const NSL::Tensor<Type> & hops) {
    NSL::Tensor<Type> adjacent(hops);
    // TODO: actually compute the adjacency: 1 if connected, 0 if not. 
    return adjacent;
}

} // namespace Lattice
} // namespace NSL
