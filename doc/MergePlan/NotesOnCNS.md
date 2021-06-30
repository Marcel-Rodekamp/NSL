# Information on CNS

CNS provides a library to simulate Hexagonal Carbon Nanostructures using the Hybrid Monte Carlo algorithm.
Its strength lay in the computational speed due to pseudo fermions.
We aim to maintain the following features:

* Pseudo fermions
    * fermion determinant replaced with pseudofermions
    * Order Nx*Ny algorithm
    * Possible on bipartite lattices for now
    * Non-explicit construction of the fermion matrix
* Forward and backwards differencing
* Iterative solvers
    * CG
    * FGMRES
* Omelyan
    * Symplectic integrator for molecular dynamics
* Observables
    * Correlators for different momenta
    * Noisy trace
