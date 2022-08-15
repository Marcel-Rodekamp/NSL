#include "NSL.hpp"

int main(){
    int nt=16, nx=2;
    NSL::Tensor<NSL::complex<double>> psi(NSL::GPU(),nt,nx), phi(NSL::GPU(),nt,nx);
    phi.rand();
    psi.rand();
    NSL::Lattice::Ring<NSL::complex<double>> lat(nx);
    lat.to(NSL::GPU());
    
    NSL::FermionMatrix::HubbardDiag M(lat,psi);
    std::cout<<M.logDetM()<<std::endl;

    return(0);
}