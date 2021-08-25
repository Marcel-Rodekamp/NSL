//Benchmarking
#include <iostream>
//hpp files from the project
#include "Tensor/tensor.hpp"
#include "LinAlg/mat_vec.hpp"
#include "FermionMatrix/fermionMatrix.hpp"
#include<torch/torch.h> //Pytorch
#include <vector> //Vector
#include<ctime> //To compute time
#include <benchmark/benchmark.h> //Benchmark

//BM
static void BM_fermionMatrix(benchmark::State & state){
    const long int Nt = state.range(0);
    const long int Nx = state.range(1);

    NSL::TimeTensor<c10::complex<double>> phi({Nt,Nx});
    NSL::Tensor<c10::complex<double>> expKappa({Nx,Nx});
    NSL::TimeTensor<c10::complex<double>> psi({Nt,Nx});
    phi.rand();
    psi.rand();
    expKappa.rand();

    for (auto _ : state){
        NSL::TestingExpDisc::exp_disc_Mp(phi, psi,expKappa);
    }
}

//Custom Argument: With this function you can control the conditions of our benchark (dimension, size per dimension)
static void CustomArguments(benchmark::internal::Benchmark* b) {
    // Nt = max t = 100
    for (int Nt = 10; Nt <= 100; Nt+=10)
        // Nx = max x = 100
        for (int Nx = 10; Nx <= 100; Nx += 10)
            b->Args({Nt, Nx});
}

BENCHMARK(BM_fermionMatrix)->Apply(CustomArguments);

//==========================================================================
//Main
BENCHMARK_MAIN();
