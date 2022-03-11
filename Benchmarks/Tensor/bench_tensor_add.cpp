#include "NSL.hpp"

#include "../benchmark.hpp"

int main(){
    // size of the tensor
    int tensor_size = 100;
    
    // Number of operation done between measurements
    int Nsweep = 10000;

    // Number of measurements
    int Nmeas = 100;

    // Define tensors which should be added
    NSL::Tensor<float> A(tensor_size); A.rand();
    NSL::Tensor<float> B(tensor_size); B.rand();

    // Store the timings in this tensor
    NSL::Tensor<double> timings(Nmeas);

    // Define the timer
    Timer<std::chrono::nanoseconds> timer("add");

    // do Nmeas timing measurements
    for(int meas = 0; meas < Nmeas; ++meas){

        // start the timer
        timer.start();

        // do Nsweep additions
        for(int sweep = 0; sweep < Nsweep; ++sweep){
            A+B;
        }

        // stop the timer
        timer.stop();

        // store the measurement
        timings(meas) = timer.get_time(1./Nmeas);
    }

    // report
    std::cout << "Vector add timings [ns]:" << std::endl
              << timings << std::endl;

}
