#ifndef NSL_LATTICE_TPP
#define NSL_LATTICE_TPP
#include "lattice.hpp"

namespace NSL::Lattice {

template <typename Type>
NSL::Lattice::SpatialLattice<Type>::SpatialLattice(
    const std::string &name,
    const NSL::Tensor<Type> &hops,
    const NSL::Tensor<double> &sites
    ) :
    name_(name),
    hops_(hops),
    // This construction for adj_ prevents the torch runtime warning "Casting complex values to real discards the imaginary part (function operator())"
    adj_(static_cast<NSL::Tensor<bool>>(NSL::LinAlg::abs(hops))),
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
NSL::Tensor<int> NSL::Lattice::SpatialLattice<Type>::adjacency_matrix(){
    return adj_;
}

template <typename Type>
NSL::Tensor<Type> NSL::Lattice::SpatialLattice<Type>::hopping_matrix(Type delta){
    return delta == static_cast<Type>(1.) ? hops_ : hops_ * delta;
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
void NSL::Lattice::SpatialLattice<Type>::compute_adjacency() {
    // This construction prevents the torch runtime warning "Casting complex values to real discards the imaginary part (function operator())"
    this->adj_ = static_cast<NSL::Tensor<int>>(static_cast<NSL::Tensor<bool>>(NSL::LinAlg::abs(this->hops_)));
}

template <typename Type>
bool NSL::Lattice::SpatialLattice<Type>::bipartite(){
    if (!(this->bipartite_is_initialized_)) this->compute_bipartite();
    return bipartite_;
}

template <typename Type>
void NSL::Lattice::SpatialLattice<Type>::compute_bipartite(){
    // We do a depth-first search for a contradiction on the adjacency graph.

    NSL::Tensor<int> adjacency = this->adjacency_matrix();
    std::queue<int> unvisited;
    for(int i=0; i < adjacency.shape(0); ++i) unvisited.push(i);

    std::map<int,int> partition;
    std::queue<int> to_visit;
    
    // In case the adjacency graph consists of different disconnected pieces,
    // just tracing down everybody's neighbors won't give a globally-correct picture.
    // Every segment must be bipartite!
    while(! unvisited.empty()){
        int head = unvisited.front();
        unvisited.pop();

        // Fast-forward to the lowest-indexed unvisited site:
        if(partition.contains(head)) continue;

        // We'll use +1 and -1 for the two partitions.
        partition[head] = 1;
        to_visit.push(head);

        while(! to_visit.empty() ){
            int current = to_visit.front();
            to_visit.pop();

            int other_partition = -1 * partition[current];

            for(int neighbor=0; neighbor<adjacency.shape(0); ++neighbor){
                if(neighbor == current) continue;
                if(adjacency(current,neighbor) == 0) continue;
                // Now we know that the current and neighbor are actually adjacent.

                if(!partition.contains(neighbor)){          // If the neighbor is new,
                    partition[neighbor] = other_partition;  // put it in the other partition
                    to_visit.push(neighbor);                // and check *its* neighbors.
                    continue;
                }

                // Now we know that the neighbor has been visited before.
                // If the neighbor's partition is correct, keep checking.
                if(partition[neighbor] == other_partition) continue;

                this->bipartite_ = false;               // Otherwise, we've found a problem,
                this->bipartite_is_initialized_ = true; // know the answer,
                return;                                 // and can stop.
            }

        }

    }

    // Since every node is visited and no contradiction was found,
    this->bipartite_ = true;                    // the graph really is bipartite!
    this->bipartite_is_initialized_ = true;     // and we don't have to worry again.
}

} // namespace NSL
#endif // NSL_LATTICE_TPP
