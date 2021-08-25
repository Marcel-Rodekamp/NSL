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
    std::vector<long int> dimension1;
    std::vector<long int> dimension2;
    dimension1.push_back(50);
    for (int i=0; i<1; i++){
        dimension1.push_back(pow(state.range(1),state.range(0)));
    }

    for(int i = 0; i < 2; i++){
        dimension2.push_back(pow(state.range(1),state.range(0)));
    }

    NSL::TimeTensor<c10::complex<double>> phi(dimension1);
    NSL::Tensor<c10::complex<double>> expKappa(dimension2);
    NSL::TimeTensor<c10::complex<double>> psi(50);
    phi.rand();
    psi.rand();
    expKappa.rand();

    for (auto _ : state){
        NSL::TestingExpDisc::exp_disc_Mp(phi, psi,expKappa);
    }
}

//Custom Argument: With this function you can control the conditions of our benchark (dimension, size per dimension)
static void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int i = 2; i <= 3; ++i)
        for (int j = 2; j <= 10; j += 1)
            b->Args({i, j});
}

BENCHMARK(BM_fermionMatrix)->Apply(CustomArguments);

//==========================================================================
//Main
BENCHMARK_MAIN();
