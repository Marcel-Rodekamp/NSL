#include <iostream>
//hpp files from the project
#include "Tensor/tensor.hpp"
#include "LinAlg/mat_vec.hpp"
#include "FermionMatrix/fermionMatrix.hpp"
#include<torch/torch.h> //Pytorch
#include <vector> //Vector
#include<ctime> //To compute time
#include <benchmark/benchmark.h> //Benchmark

//int main() {
//Examples for tensor-TimeTensor class.
    //Construction.
    /*NSL::Tensor<c10::complex<float>> example_tensor0(2); //Vector
    NSL::Tensor<c10::complex<float>> example_tensor1({2,2,2}); //Tensor
    NSL::TimeTensor<c10::complex<double>> example_timetensor0 (2); //Time-vector
    NSL::TimeTensor<c10::complex<double>> example_timetensor1 ({2,2,2}); //Time-tensor
     */

    //Copy.
    /*NSL::Tensor<c10::complex<float>> example_tensor2 (2); //Initialize tensor
    NSL::Tensor<c10::complex<float>> example_tensor3(example_tensor2); //Copy of tensor
    NSL::TimeTensor<c10::complex<float>> example_timetensor2(2); //Initialize Time-tensor
    NSL::TimeTensor<c10::complex<float>> example_timetensor3(example_tensor2); //Copy of Time-tensor
     */

    //SHAPE.
    /*NSL::Tensor<c10::complex<float>> example_tensor4 ({2,2,2}); //Initialize tensor
    std::cout<<example_tensor4.shape(1)<<std::endl; //Print shape of input position in terminal
    NSL::TimeTensor<std::complex<float>> example_timetensor4({60,60,12}); //Initialize time-tensor
    std::cout<<example_tensortime4.shape(2)<<std::endl; //Print shape of input position in terminal
     */

    //random access.
    /* NSL::Tensor<float> example_tensor5({2,2});
     example_tensor5[{1,1}]=1;
     NSL::TimeTensor<float> example_timetensor5({2,2});
     example_timetensor5[{1,1}]=1;*/

    //Example of the use of operator *
    /*NSL::Tensor<double> example_tensor6({2,2});
    (example_tensor4*3);
     NSL::TimeTensor<double> example_timetensor6({2,2});
    (example_timetensor6*2);*/

    //Example of the use of exponential
    /*NSL::Tensor<std::complex<float>> example_tensor7 ({3,3}); //Initialize tensor
    example_tensor5.exp();
    NSL::TimeTensor<std::complex<float>> example_timetensor7 ({3,3}); //Initialize tensor
    example_timetensor7.exp();*/

    //Example of expand
    /*NSL::Tensor<std::complex<float>> example_tensor8({3, 2});
    std::deque<long int> dimension8 ={3, 2, 2};
    example_tensor8.expand(dimension8))
    NSL::TimeTensor<std::complex<float>> example_timetensor8({3, 2});
    example_timetensor8.expand(dimension8));*/

    //Tensor shift
    /*NSL::Tensor<c10::complex<double>> example_tensor9 ({2,2});
    c10::complex<double> num(3,2);
    example_tensor9.shift(1, num);
    NSL::TimeTensor<c10::complex<double>> example_timetensor9 ({2,2});
    example_timetensor9.shift(1, num);*/

//===========================================================================
//Example of mat_vec.hpp. Must change data_ from private //
    //Example of mat_vec Tensor x TimeTensor
    /*NSL::Tensor<c10::complex<float>> example_tensor20 ({2,2});
    NSL::TimeTensor<c10::complex<float>> example_timetensor20 ({2,2});
    NSL::LinAlg::mat_vec(example_tensor20, example_timetensor20); //Tensor x TimeTensor
    NSL::LinAlg::mat_vec(example_timetensor20, example_tensor20); //TimeTensor x Tensor
    NSL::LinAlg::mat_vec(example_tensor20, example_timetensor20); //TimeTensor x TimeTensor
    NSL::LinAlg::mat_vec(example_tensor20, example_tensor20);     //Tensor x Tensor
     */

    //Exponential
    /*NSL::Tensor<double> example_tenso21({2,2});
     NSL::TimeTensor<double> example_timetensor21({2,2});
     NSL::LinAlg::exp(example_tensor21);
     NSL::LinAlg::exp(example_timetensor21);*/

    //Expand
    /* NSL::Tensor<double> example_tensor22({2,2});
     NSL::TimeTensor<double> example_timetensor22({2,2});
     std::deque<long int> expand_num22 ={2};
     NSL::LinAlg::expand(example_tensor22, expand_num22);
     NSL::LinAlg::expand(example_timetensor22, expand_num22);
     */

    //Shift
    /*NSL::Tensor<double> example_tensor23({2,2});
    NSL::TimeTensor<double> example_timetensor23({2,2});
    NSL::LinAlg::shift(example_tensor23, 1, 3.0);
    NSL::LinAlg::shift(example_timetensor23, 1, 3.0);
    */

//==================================================================================
//Example of FermionMatrix
    //Example of BF
/*    NSL::TimeTensor<c10::complex<double>> phi({2,2});
    NSL::Tensor<c10::complex<double>> expKappa({2,2});
    phi.print();
    expKappa.print();
    (NSL::TestingExpDisc::BF(phi, expKappa)).print();
    phi.print();
    expKappa.print();*/
//==================================================================================
//return 0;
//} //end of int main
//===============================================================================
//===============================================================================
//Benchmarking
//BM
static void BM_fermionMatrix(benchmark::State & state){
    std::vector<long int> dimension;
    for(int i = 0; i < state.range(0); i++){
        dimension.push_back(state.range(1));
    }
    NSL::TimeTensor<c10::complex<double>> phi(dimension);
    NSL::Tensor<c10::complex<double>> expKappa(dimension);
    for (auto _ : state){
        NSL::TestingExpDisc::BF(phi, expKappa);
    }
}
BENCHMARK(BM_fermionMatrix)
    ->Args({2,20})
    ->Args({2,40})
    ->Args({2,60})
    ->Args({2,80})
    ->Args({2,100})
    ->Args({3,20})
    ->Args({3,40})
    ->Args({3,60})
    ->Args({3,80})
    ->Args({3,100}); //Register the function as a benchmark
//==========================================================================

//Main
BENCHMARK_MAIN();
