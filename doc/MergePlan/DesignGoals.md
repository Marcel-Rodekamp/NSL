# Design Goals

This document provides a general idea of the software design.

## During Merge

* Hybrid code:
    * Multi Node
        * Possible domain decomposition of temporal dimension
        * Feature will be added later, empty interface during merge
    * Multi CPU
        * Multiple Processes should be usable
        * Adding GPUs:
            * 1 Process per GPU
    * Multi GPU:
        * Nvidia
        * AMD
            * If required edit interfaces after merge
    * Don't reinvent the wheel
        * Library for memory management:
        * Library for algebra management:
        * Library for file IO based on h5py:
        * Library for communication (Multi node): Comes later
* Interface
    * Usability from C++ and Python
    * Default: Expose every class/function to python
* Interfacing
    * NumPy
    * Pytorch (libtorch)
* Unit Tests
    * Catch2 (C++ site)
    * unittest (python)
* Portability
    * Linux (tests, development, production)
    * Mac Os (tests, development)

### Interfaces

Hide underlying libraries in interfaces.
Allowing to interchange libraries in case other features are required.
From high to low level, consider the following workflow for an HMC call.
Note that this is not detailed documentation of the classes!

1. `NSL::HMC`
    * Classes implementing a HMC algorithm given
        * Integrator
        * Action
        * HMC parameters

```
// list of configs
using NSL::Ensemble = std::deque<NSL::ConfigBase>

class MarkovChain{
    private:
        NSL::Ensemble MC;

        ProposalMachineBase * pmb;

    public:
        NSL::ConfigBase & get_recent_config(){
            return *(MC.end()-1);
        }

        // call operator() of pmb and store config at end of MC
        void update();
        // call update N_MC times
        void generate_ensemble(size_t N_MC);

        // return the idx^th config
        NSL::ConfigBase operator[](size_t idx){
            return MC[idx];
        }

        // std access function
};

class ProposalMachineBase{
    std::list<NSL::RNG &> rng;

    public:
        // Computes config + Accept Reject
        virtual NSL::ConfigBase & operator()(NSL::ConfigBase * in_config) = 0;
}

class HMC: public ProposalMachineBase {
    private:
        // omelyan, leapfrog
        NSL::IntegratorBase * integrator;

        // Action
        NSL::ActionBase * action;

        // Parameters used during HMC
        NSL::ParameterDict * param;

    public:
        // make up momenta
        // integrate EoM
        // compute action difference
        // accept reject
        // return result
        NSL::ConfigBase & operator()(NSL::ConfigBase * config)
};
```
2. `NSL::Integrator`
    * Classes implementing different integrators given
        * Action
        * Fermion Matrix
        * Integrator parameters

```
class FieldIntegratorBase{
    private:
        // Fermion Matrix
        NSL::FermionMatrixBase * FermMat;  

        // Action
        NSL::ActionBase * action;

        // parameters used during integration
        NSL::ParameterDict * param;

    public:
        // integrates
        NSL::ConfigBase * operator()(NSL::ConfigBase *, size_t steps = 1);
};
```
3. `NSL::Action`
    * Classes implementing different actions given
        * Fermion Matrix
        * Action parameters

```
class ActionBase{
    private:
        // parameters used during action
        NSL::ParameterDict * param;

    public:

    Complex operator()(NSL::ConfigBase * config);
    NSL::ConfigBase * force(NSL::ConfigBase * config);
    NSL::ConfigBase * grad(NSL::ConfigBase * config);
}
```
4. `NSL::FermionMatrix`
    * Classes implementing certain fermion matrices given
        * Lattice
        * Hopping matrix
        * Fermion Matrix Parameters

```
class FermionMatrix{
    private:
        // encodes nearest neighbor table
        NSL::LatticeBase * lattice;

        // parameters used during Fermion Matrix
        NSL::ParameterDict * param;



    public:

        FermField M(...);
        FermField dM(...);
        FermField Mdag(...);
        FermField dMdag(...);
        FermField MdagM(...);
        FermField dMdagM(...);
};
```
5. `NSL::Lattice`
    * Implements different spatial lattices and communication of temporal axis given

```
struct Edge{
    size_t s_start,s_finish;
    // hopping strength from s_start,s_finish
    Complex k;
};

class NSL::SpatialLatticeBase{
    private:
        // Topology only
        // | Site Index | Nearest Neighbor List | kappa (hopping strength)   |
        // |      0     | 1,2,V, ...            | E(0,1), E(0,2), E(0,V),... |
        // |      1     | 0,2,3,V-1 ...         | E(1,0), E(1,2), E(1,V),... |
        // SpatialLattice[0] = Site
        // SpatialLattice(0,0) = Hopping strength 0 - 0
        std::vector<std::vector<Edge>> hopping_graph;
        std::vector<NSL::Site> linear_index_trafo;

        // Hopping matrix
        NSL::Tensor & Hopping_matrix;

    public:
        // Return linearized index from lattice site
        size_t operator()(NSL::Site & lattice_site);

        // Return lattice site from linear index
        NSL::Site & operator()(size_t index);

};

template<size_t dim>
__device__ __host__ Site = Array<dim>;
```
6. `NSL::FermField`
    * Implements a (pseudo-)fermion field given
        * Lattice

```
class FermField : NSL::Tensor {
    private:
        NSL::LatticeBase * lattice;

        Complex Action_Value;

    public:
        // Tensor operations but agnostic to lattice
}
```
7. `NSL::Tensor`
    * Implements the memory and algebra interface

```
class Tensor {
    private:
        //library container
        Container * data;

    public:
        // Interface all relevant functions
        // 1. Elementwise
        // 2. Tensor - Tensor operations
}
```
8. `NSL::LinearAlgebra::`
    * Namespace of functions performing algorithms on `NSL::Tensor`

## After Merge
