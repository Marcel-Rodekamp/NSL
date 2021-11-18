#include "../benchmark.hpp"
#include "FermionMatrix/fermionMatrixHubbardExp.hpp"
#include "Lattice/Implementations/ring.hpp"
#include <iostream>
#include <fstream>
using size_type = int64_t;

auto benchmark_fermionMatrixHubbardExp(const size_type Nt,const size_type Nx, std::string fn = "benchmark.dat"){

    std::cout << "Benchmarking FermionMatrixHubbardExp::M";

    const size_type Nmeas = 100;

    std::ofstream fout(fn);
    
    NSL::TimeTensor<NSL::complex<double>> phi(Nt,Nx);
    NSL::TimeTensor<NSL::complex<double>> psi(Nt,Nx);
    phi.rand();
    psi.rand();

    NSL::Lattice::Ring<double> r(Nx);
    NSL::FermionMatrix::FermionMatrixHubbardExp M(&r, phi);

    Timer<std::chrono::high_resolution_clock, std::chrono::nanoseconds> timer("FermionMatrixHubbardExp.M");

    timer.start();
    for(int n = 0; n < Nmeas; ++n){
        psi = M.M(psi);
    }
    timer.stop();

    double time = timer.get_time(1e-9/Nmeas);

    fout <<  time << std::endl;

    fout.close();

    // don't remove this, it is calling psi such that the compiler does not optimize this away
    std::cout << "(Nt=" << psi.shape(0)
              << ",Nx=" << psi.shape(1) << "): "
              << time << " s" << std::endl;

}

int main(){
    benchmark_fermionMatrixHubbardExp(16,2, "bm_hubbardFermionMatrixExp_M_Nt16_Nx2.dat");

    return EXIT_SUCCESS;
}