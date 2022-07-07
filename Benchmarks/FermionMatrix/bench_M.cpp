#include "NSL.hpp"
#include <cmath>

#include "../benchmark.hpp"
#include "FermionMatrix/Impl/hubbardDiag.hpp"
#include "FermionMatrix/Impl/hubbardDiag.tpp"

int main(){
    // size of the tensor
    int tensor_size = 100;
    
    // Number of operation done between measurements
    int Nsweep = 1000;

    // Number of measurements
    int Nmeas = 100;

    // Mean time and estimated bandwidth
    double mean_time=0.0, est_bw;

    // Define the necessary Tensors and objects
    int nt = 16, nx = 2;
	NSL::Tensor<NSL::complex<double>> phi(nt, nx);
    NSL::Tensor<NSL::complex<double>> psi(nt, nx);
  
    phi.rand();
    psi.rand();

    NSL::Lattice::Ring<NSL::complex<double>> lat(nx);
    NSL::FermionMatrix::HubbardDiag  M(lat,phi);

    //Memory in GB
    const double mem_GB  = (3 * (2*nt*nx * sizeof(NSL::complex<double>)) + nt*nx*(nx) * sizeof(NSL::complex<double>)) * pow(10,-9); //9.3 * pow(10,-10);

    // Store the timings in this tensor
    NSL::Tensor<double> timings(Nmeas);

    // Define the timer
    Timer<std::chrono::nanoseconds> timer("add");

    // do Nmeas timing measurements
    for(int meas = 0; meas < Nmeas; ++meas){

        // start the timer
        timer.start();

        // call M Nsweep times
        for(int sweep = 0; sweep < Nsweep; ++sweep){
            M.M(psi);
        }

        // stop the timer
        timer.stop();

        // store the measurement
        timings(meas) = timer.get_time(1./Nsweep);
    }

    // mean time = total time/number of measurements
    for(int meas =0; meas < Nmeas; ++meas){
        mean_time = mean_time + timings(meas);
    }
    mean_time = (mean_time/Nmeas) * pow(10,-9);

    //Estimated bandwidth
    est_bw =  mem_GB/mean_time ;

    // report
    std::cout << "Estimated Bandwidth [GB/s]: "<<est_bw<<std::endl;

}
