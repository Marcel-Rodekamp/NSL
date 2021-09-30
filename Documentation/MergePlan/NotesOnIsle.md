# Information on Isle

[Isle](https://github.com/evanberkowitz/isle) provides a library to simulate Carbon Nanostructures on various lattices using the Hybrid Monte Carlo algorithm.
Its strength lay in its high expressibility as well as a simple interface.
We aim to maintain the following features:

* Python interface
    * Algorithms implemented in C++
    * Algorithms exposed to python using pybind11
* Expressivity to many systems
    * bipartite and non-bipartite lattices
    * chemical potential
    * Available through direct diagonalization
        * order Nx³ * Nt
* Modularity
    * Everything is expressed via a class template
* Interoperability with NumPy
    * Vector class can handle NumPy arrays coming from python
* Leapfrog
    * Symplectic integrator for the molecular dynamics step
* Runge Kutta
    * Integrator used in Holomorphic flow
* File IO & Logging
    * Python site
    * h5 format
* Different discretizations
    * Diagonal
    * Exponential
* Observables
    * All to All propagator
