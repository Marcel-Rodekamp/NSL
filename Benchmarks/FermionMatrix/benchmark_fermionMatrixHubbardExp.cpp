#include <chrono>
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"
#include "Lattice/Implementations/ring.hpp"
#include <iostream>
using size_type = int64_t;

void benchmark_fermionMatrixHubbardExp(const size_type Nt,const size_type Nx){
    const size_type Nmeas = 100;
    std::ofstream fout("benchmark_fermionMatrixHubbardExp_data.txt", std::ios::out | std::ios::trunc | std::ios::app | std::ios::binary);
    
    NSL::TimeTensor<NSL::complex<double>> phi(Nt,Nx);
    NSL::TimeTensor<NSL::complex<double>> psi(Nt,Nx);
    phi.rand();
    psi.rand();

    const size_type num_ele = Nt*Nx;
    const size_type mem = 2 * num_ele * sizeof(NSL::complex<double>); //2 for phi and psi
    
    NSL::Lattice::Ring<double> r(Nx);

    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r, phi);

    auto start = std::chrono::high_resolution_clock::now();
    // loop over it to get some statistical significance and average out caching effects!
    for(int n = 0; n < Nmeas; ++n){
        psi = M.M(psi);
    }
    auto end = std::chrono::high_resolution_clock::now();
  
    std::cout << "psi[0] = " << psi(0,0) << std::endl;

    double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  
    time_taken *= 1e-9/Nmeas;

    std::cout<<"Dimensions : " << Nt << " x " <<Nx<<std::endl;
    std::cout<<mem<<std::endl;
    std::cout << "Time taken by program is : " 
         << time_taken << std::setprecision(9);
    std::cout << " sec" << std::endl;
    
    fout << time_taken << std::endl;

    fout.close();
}

int main(){
    benchmark_fermionMatrixHubbardExp(16,2);
    return EXIT_SUCCESS;
}