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


## After Merge
