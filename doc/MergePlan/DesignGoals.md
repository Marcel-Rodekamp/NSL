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

#### 1. Ensamble generation
* Markov Chain:
    * Store configurations
    * Interface to generate 1 or many configurations

```
// list of configs
using NSL::Ensemble = std::deque<NSL::ConfigBase>

template<class ConfigType, class ProposalMachineType>
class MarkovChain{
    private:
        NSL::Ensemble _MC;
        ProposalMachineType * _pmb;

    public:
        // constructors
        // construct MarkovChain with Hot Start
        MarkovChain();
        // Construct MarkovChain with given start config
        explicit MarkovChain(ConfigType&&);
        // Construct MarkovChain from range of configurations
        template<class InputIt> MarkovChain(InputIt first, InputIt last);
        // Copy constructor
        MarkovChain(const MarkovChain& other);
        // Move constructor
        MarkovChain(MarkovChain&& other);

        // Replace contents of the class
        void assign( size_type count, const Configtype& MC, ProposalMachineType & pmb)
        template< class InputIt > void assign( InputIt first, InputIt last, ProposalMachineType & pmb );

        // call operator() of pmb to generate a new configuration
        // Store it at end of MC
        void update();

        // call update() N_MC times
        void generate_ensemble(size_type N_MC);

        // get pos^th config
        reference at( size_type pos );
        const_reference at( size_type pos ) const;

        // get pos^th config
        reference operator[]( size_type pos );
        const_reference operator[]( size_type pos ) const;

        // get first config of the chain
        reference front();
        const_reference front() const;

        // get last config of the chain
        reference back();
        const_reference back() const;

        // get iterator to the first config
        iterator begin() noexcept;
        const_iterator begin() const noexcept;
        const_iterator cbegin() const noexcept;

        // get iterator past the last config
        iterator end() noexcept;
        const_iterator end() const noexcept;
        const_iterator end() const noexcept;

        // check if the markov chain is empty
        bool empty() const;
        [[nodiscard]] bool empty() const noexcept;

        // return number of configs in the markov chain
        size_type size() const noexcept;

        // return the maximum number of configurations that can be stored
        size_type max_size() const noexcept;

        // Requests the removal of unused capacity.
        void shrink_to_fit();

        // erase all configs
        void clear() noexcept;

        // inserts a configuration
        iterator insert( const_iterator pos, const T& value );
        iterator insert( const_iterator pos, T&& value );
        iterator insert( const_iterator pos, size_type count, const T& value );
        template< class InputIt > iterator insert( const_iterator pos, InputIt first, InputIt last );
        iterator insert( const_iterator pos, std::initializer_list<T> ilist );

        // inserts a config directly before pos
        template< class... Args > iterator emplace( const_iterator pos, Args&&... args );

        // eases a config from the chain
        iterator erase( const_iterator pos );
        iterator erase( const_iterator first, const_iterator last );

        // Appends a confuguration to the chain
        void push_back( const T& value );
        void push_back( T&& value );

        // Appends a new configuration to the end of the chain
        template< class... Args > reference emplace_back( Args&&... args );

        // removes last configuration
        void pop_back();

        // prepends a given configuration to the beginning of the chain
        void push_front( T&& value );

        // Appends a new configuration to the beginning of the chain
        template< class... Args > reference emplace_front( Args&&... args );

        // removes the first element from thecontainer
        void pop_front();

        // resizes the chain
        void resize( size_type count );
        void resize( size_type count, const value_type& value );

        // exchanges the markov chain with another
        void swap( MarkovChain& other ) noexcept(std::allocator_traits<Allocator>::is_always_equal::value);
};
```

* Proposal machines:
    * Base class to update configs
    * Holds RNG state

```
class ProposalMachineBase{
    std::list<NSL::RNG &> _rng;

    public:
        // Computes config + Accept Reject
        virtual NSL::ConfigBase & operator()(NSL::ConfigBase * in_config) = 0;

        // access the RNG(s)
        NSL::RNG & get_rng(size_type idx = 0);
}
```

* Examples Proposal Machine: HMC
    * Holds on the integrator it uses for molecular dynamics
    * Holds on the action
    * Uses the parameter dict to access
        * `MD Trajectory Length`
        * `Number of MD steps`
        * -> ceck that `MD Trajectory Length / Number of MD steps` matches `MD Step Size`

```
template<class IntegratorType, ckass ActionType, class ConfigType>
class HMC: public ProposalMachineBase {
    private:
        // omelyan, leapfrog, ...
        IntegratorType & _integrator;

        // Action
        ActionType & _action;

        // Parameters used during HMC
        NSL::ParameterDict & _param;

        // store old action for accept reject
        Complex _Sold = 0;

    public:
        HMC() = delete;
        // constructor
        HMC(IntegratorType & integrator, ActionType & action, NSL::ParameterDict & param);
        // copy constructor
        HMC(HMC& other);
        // move constructor
        HMC(HMC&& other);


        // make up momenta
        // integrate EoM
        // compute action difference
        // accept reject
        // return result
        ConfigType & operator()(ConfigType & config);

        // compute the hamiltonian p^2 + action(config)
        Complex computeHamilton(ConfigType & momentum, ConfigType & config);
};
```

#### 2. Integrators

* FieldIntegrators
    * Algorithm to integrate EoM regarding configurations
        * Holds on the action
        * Holds on the fermion Matrix
        * Integrator parameters
            * step size

```
template<class ActionType, class FieldType>
class FieldIntegratorBase{
    private:
        // Action
        ActionType & _action;

        // parameters used during integration
        NSL::ParameterDict & _param;

    public:
        // integrates
        virtual FieldType operator()(FieldType start, size_t steps = 1) = 0;
};
```

#### 3. Actions
* Implementing different actions
    * Holds on the fermion matrix
    * Holds on the action parameters
        * Coubling strength
    * Provides `operator()` computing the acion of a given config
    * Provides `eval` being the same as `operator()`
    * Provides `force` computing the force of a given config
    * Provides `grad` providing the derivative of the action in respect to a given config

```
template<class ConfigType>
class ActionBase{
    private:
        // parameters used during action
        NSL::ParameterDict & _param;

    public:
        virtual Complex operator()(ConfigType & config) = 0;
        virtual Complex eval(ConfigType & config) = 0;
        virtual ConfigType & force(ConfigType & config) = 0;
        virtual ConfigType & grad(ConfigType & config) = 0;
}
```

#### 4. Simulation Objects

* Fermion Matrix
    * Implementing different diskretizations
    * Implementing different algorithms
        * Matrix Vector / Explicit construction
            * `M`,`Mdag`,`MMdag`,`MdagM`
            * Derivatives
            * `det_M`,`det_Mdag`,`det_MdagM`,`det_MMdag`
            * `log_det_M`,`log_det_Mdag`,`log_det_MdagM`,`log_det_MMdag`
        * open question: How to design?
            * Inheritence
            * Class templates

```
template<class FieldType, class Algorithm>
class FermionMatrixBase{
    private:
        // encodes nearest neighbor table
        NSL::LatticeBase * _lattice;

        // parameters used during Fermion Matrix
        NSL::ParameterDict * _param;

    public:

        virtual FieldType & M(FieldType & config) = 0;
        virtual FieldType & Mdag(FieldType & config) = 0;
        virtual FieldType & MdagM(FieldType & config) = 0;
        virtual FieldType & MMdag(FieldType & config) = 0;

        virtual Complex det_M(FieldType & config) = 0;
        virtual Complex det_Mdag(FieldType & config) = 0;
        virtual Complex det_MdagM(FieldType & config) = 0;
        virtual Complex det_MMdag(FieldType & config) = 0;

        virtual Complex log_det_M(FieldType & config) = 0;
        virtual Complex log_det_Mdag(FieldType & config) = 0;
        virtual Complex log_det_MdagM(FieldType & config) = 0;
        virtual Complex log_det_MMdag(FieldType & config) = 0;
};
```

* Fields
    * Represent fileds in classes derived from this
    * they must contain information about
        * Space Time
        * data
        * Time Slices
            - Return Spatial Lattice at a certain time point
        * Fibres
            - Return part of spatial lattice for all time points

```
template<class SpaceTimeLatticeType>
class FieldBase {
    private:
        SpaceTimeLatticeType & _spaceTimeLattice;
        NSL::Tensor _data;

    public:
    FieldBase(size_type nt, size_type nx);

    // maybe use NSL::TensorView
    NSL::Tensor & space_time_vector();

    // return an spatial volume tensor
    NSL::Tensor & time_slice(size_type t);

    //
    NSL::Tensor & fibre(size_type x);
}

```

#### 5. Space Time Information

* LatticeClasses
    * Encode information about spatial and temporal lattice

* Edge
    * representing the hopping from a spatial site to another

```
struct Edge{
    size_t s_start,s_finish;
    // hopping strength from s_start,s_finish
    Complex k;
};

```
* Spatial Lattice
    - Spatial Topology
    - Graph representing the hopping
    - Hopping matrix
    - Exponential of hopping matrix
    - Spatial Volume

```

class SpatialLatticeBase: {
    private:
        // Topology only
        // | Site Index | Nearest Neighbor List | kappa (hopping strength)   |
        // |      0     | 1,2,V, ...            | E(0,1), E(0,2), E(0,V),... |
        // |      1     | 0,2,3,V-1 ...         | E(1,0), E(1,2), E(1,V),... |
        std::vector<std::vector<Edge>> _hoppingGraph;
        std::vector<NSL::Site> _linearIndexTrafo;

        // Hopping matrix
        NSL::Tensor & _hoppingMatrix;

        const size_type _volume;

    public:
        // Return linearized index from lattice site
        size_t operator()(NSL::Site & lattice_site);

        // Return lattice site from linear index
        NSL::Site & operator()(size_t index);

        NSL::Tensor hopping_matrix(){
            return _hoppingMatrix;
        }

        //
        NSL::Tensor exp_hopping_matrix(){
            return _hoppingMatrix.matexp();
        }

};
```

* Space Time Lattice
    - contains spatial lattice object
    - Temporal axis
    - open question: Is it aware of the communicator

```
template<SpatialLatticeType>
class SpaceTimeLatticeBase{
    private:
        const SpaceTimeLattice & _spatLattice;
        const size_type _nt;
        NSL::ParameterDict _param{}

        // later ?
        NSL::Communicator _comm;
    public:
        SpaceTimeLatticeBase(spatLattice,param,nt){...}
        SpaceTimeLatticeBase(spatLattice,param){
            _nt = param.nt
        }

        template<typename Args...>
        SpaceTimeLatticeBase(param,nt,Args... spatLatArgs)
};


template<size_t dim>
__device__ __host__ Site = Array<dim>;
```

#### Linear Algebra ToDo

* linear algebra interface
    * Allows to interface with different libraries
    * extendability
    * collection of functions/classes
* Tensor interface
    * memory management
    * works with the linAlg interface
    * one class template exposes everything
    * Full STL like accessor methods
        * we do not need to be constructor aware do we?
        * GPU agnostic
    * View/Accessor arithmetic
        * CPU-GPU in one interface ?
    * Pytorch like linear algebra exposure
        * e.g. Tensor.exp() Elementwise exp
        * e.g. Tensor.mat_exp() Matrix exponential
    * Exportability to python via pybind11
    * Handling data access from numpy and pytorch (libtorch)
        * either copy/move constructor
        * way to export (copy/move) to numpy/pytorch
    * Extenability to AMD GPUs
        * Open question

```
template<typename Type>
class Tensor {
    private:
        //library container
        // which container to choose?
        Container * data;

    public:

}
```

## Remainder

* File IO not covered
* Parameter not covered
* Benchmarking

## After Merge

## Style Guide

* classes: MyClass
    * camelCase
    * First letter capitalized
* private variables: _myVariable
    * camelCase
    * Starts with underscore
    * Second letter lower case
* functions: my_function
    * snake_case
    * First letter lower case
* function variables:
    * camelCase
    * First letter lower case
* global constants: MY_GLOBAL
    * snake_case
    * CAPITALIZED
* variables: myVariable
    * camelCase

* Tab = 4 spaces (Soft Tab)
* dont use /**/ within code
