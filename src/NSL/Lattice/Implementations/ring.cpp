#ifndef NSL_LATTICE_RING_CPP
#define NSL_LATTICE_RING_CPP

#include "ring.hpp"

template <Type>
NSL::Lattice::Ring<Type>(const unsigned int n): 
    n_(n),
    sites(sites_(n)),
    hops(hops_(n)),
    name(name_(n))
{
}

template <Type>
static std::vector<Site> & NSL::Lattice::Ring<Type>::sites_(const unsigned int n) {
    std::vector<Site> all(n);
    return all;
}

template <Type>
static NSL::Tensor<Type> & NSL::Lattice::Ring<Type>::hops_(const unsigned int n, const T kappa=1.) {
    NSL::Tensor<Type> amplitude(n);
    for (int i = 0; i < n-1; ++i ){
        amplitude(i,i+1) = kappa;
    }
    for (int i = 1; i < n;   ++i ){
        amplitude(i-1,i) = kappa;
    }
    amplitude(0,n-1) = kappa;
    amplitude(n-1,0) = kappa;
}

template <Type>
static std::string & NSL::Lattice::Ring<Type>::name_(const unsigned int n) {
    return sprintf("ring(%i)", n);
}

NSL::Lattice::Ring<float>;
NSL::Lattice::Ring<double>;

#endif
