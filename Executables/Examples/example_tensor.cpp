#include <iostream>
//hpp files from the project
#include "Tensor/tensor.hpp"
#include "LinAlg/mat_vec.hpp"
#include "FermionMatrix/fermionMatrix.hpp"
//Pytorch
#include<torch/torch.h>
//Vector
#include <vector>
//To compute time
#include<ctime>
//Benchmark
#include <benchmark/benchmark.h>


int main() { // test you tensors here if you like ;)

//Examples for tensor class.
    //Construction.
    /*NSL::Tensor<c10::complex<float>> example_tensor0 ({2});
    example_tensor0.print();
    NSL::Tensor<c10::complex<float>> example_tensor2 ({2,2,2});
    example_tensor2.print_complex();*/

    //Copy a tensor.
    /*NSL::Tensor<c10::complex<float>> example_tensor8 ({2});
    NSL::Tensor<c10::complex<float>> example_tensor9 (example_tensor8);
    example_tensor9.exp().print();
    example_tensor8.print();
*/
    //ToDo: Why better size_t than long int.
    //Example of the use of SHAPE.
    /*NSL::Tensor<c10::complex<float>> example_tensor3 ({2,2,2});
    std::size_t num = 1;
    std::cout<<"shape_value="<<example_tensor3.shape(1)<<std::endl;*/

    //Example of the use of random access.
   /* NSL::Tensor<float> example_tensor1({2,2});
    example_tensor1[{1,1}]=1;
    example_tensor1.print();*/

    //Example of the use of operator *
    /*NSL::Tensor<double> example_tensor4({2,2});
    (example_tensor4*3).print();*/

    //Example of the use of exponential
    /*   NSL::Tensor<std::complex<float>> example_tensor5 ({3,3}); //Initialize tensor
       std::cout<< example_tensor5.exp().data_<<std::endl;*/

    //Example of expand
    /*NSL::Tensor<std::complex<float>> example_tensor32({3, 2});
    std::cout << example_tensor32.data_ << std::endl;
    std::deque<long int> dimension32 ={3, 2, 2};
    std::cout << (example_tensor32.expand(dimension32)).data_ << std::endl;*/

    //Tensor shift
    /*NSL::Tensor<c10::complex<double>> example_tensor18 ({2,2});
    example_tensor18[{1}]+=1;
    c10::complex<double> num(3,2);
    example_tensor18.print();
    example_tensor18.shift(1, num).print();*/
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
// Example of Time Tensor. Must change data_ from private to public.
    //Example for Timetensor construction, 1D, 3D and copy.
    /*NSL::TimeTensor<c10::complex<double>> example_tensor20 (12);
    std::cout<< example_tensor20.data_ << std::endl;
    NSL::TimeTensor<c10::complex<double>> example_tensor21 = example_tensor20;
    std::cout<< example_tensor21.data_ << std::endl;
    std::vector<long int> dimension = {60,60,12};
    std::cout<< example_tensor20.data_ << std::endl;*/

    //Copy a tensor
    /*NSL::TimeTensor<c10::complex<float>> example_tensor28 ({2});
    NSL::TimeTensor<c10::complex<float>> example_tensor29 (example_tensor8);
    example_tensor29.exp().print();
    example_tensor28.print();*/

    //Example of the use of SHAPE.
    /*NSL::TimeTensor<std::complex<float>> example_tensor23({60,60,12});
    std::cout<<example_tensor23.shape(1);*/

    //Example of the use of random access.
    /*NSL::Tensor<c10::complex<float>> example_tensor24 ({2,2});
    example_tensor24[{0}] += 1.0;
    example_tensor24.print();*/

    //Example of the use of exponential
  /*  NSL::TimeTensor<c10::complex<float>> example_tensor25 ({2,2});
    example_tensor25.exp().print();*/

    //Example of the use of expand
     /*NSL::TimeTensor<double> example_tensor27({2,2});
     std::deque<long int> expand_number = {2};
     std::cout<< example_tensor27.expand(expand_number).data_<<std::endl;*/

    //Tensor shift
    /*NSL::TimeTensor<c10::complex<double>> example_tensor28 ({2,2});
    example_tensor28[{1}]+=1;
    c10::complex<double> num(3,2);
    example_tensor28.print();
    example_tensor28.shift(1, num).print();*/
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
//Example of mat_vec.hpp. Must change data_ from private //        Type &operator[]std::size_t idx);to public.
//Example of mat_vec Tensor x Tensor
    /*NSL::Tensor<c10::complex<float>> example_tensor30 ({2,2});
    NSL::Tensor<c10::complex<float>> example_tensor31 ({2,2});
    NSL::LinAlg::mat_vec(example_tensor30, example_tensor31).print();*/

    //Example of mat_vec Tensor x TimeTensor
    /*NSL::Tensor<c10::complex<float>> example_tensor32 ({2,2});
    NSL::TimeTensor<c10::complex<float>> example_tensor33 ({2,2});
    NSL::LinAlg::mat_vec(example_tensor32, example_tensor33).print();*/

    //Example of mat_vec TimeTensor x TimeTensor
    /*NSL::TimeTensor<c10::complex<float>> example_tensor34 ({2,2});
    NSL::TimeTensor<c10::complex<float>> example_tensor35 ({2,2});
    NSL::LinAlg::mat_vec(example_tensor34, example_tensor35).print();*/

    //Example of mat_vec TimeTensor x Tensor
    /*NSL::TimeTensor<c10::complex<float>> example_tensor36 ({2,2});
    NSL::Tensor<c10::complex<float>> example_tensor37 ({2,2});
    NSL::LinAlg::mat_vec(example_tensor36, example_tensor37).print();*/
//-------------------------------------------------------------------------------
//Exponential
    //Exponential Tensor
    /*NSL::Tensor<double> example_tensor38({2,2});
    auto example_tensor39 = NSL::LinAlg::exp(example_tensor38);
    example_tensor38.print();
    example_tensor39.print();*/

    //Exponential TimeTensor
    /*NSL::TimeTensor<double> example_tensor311({2,2});
    auto example_tensor312 = NSL::LinAlg::exp(example_tensor311);
    example_tensor311.print();
    example_tensor312.print();*/
//-------------------------------------------------------------------------------
//Expand
    //Expand Tensor
   /* NSL::Tensor<double> example_tensor313({2,2});
    std::deque<long int> expand_num314 ={2};
    auto example_tensor314 = NSL::LinAlg::expand(example_tensor313, expand_num314);
    example_tensor314.print();*/

    //Expand TimeTensor
    /*NSL::TimeTensor<double> example_tensor315({2,2});
    std::deque<long int> expand_num316 ={2};
    auto example_tensor316 = NSL::LinAlg::expand(example_tensor315, expand_num316);
    example_tensor315.print();
    example_tensor316.print();*/
//-------------------------------------------------------------------------------
//Shift
    //Shift Tensor
    /*NSL::Tensor<double> example_tensor317({2,2});
    example_tensor317[{1}] += 1;
    auto example_tensor318 = NSL::LinAlg::shift(example_tensor317, 1, 3.0);
    example_tensor317.print();
    example_tensor318.print();*/

    //Shift TimeTensor
    /*NSL::Tensor<double> example_tensor319({2,2});
    example_tensor319[{1}] += 1;
    auto example_tensor320 = NSL::LinAlg::shift(example_tensor319, 1, 3.0);
    example_tensor319.print();
    example_tensor320.print();*/
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
//Example of FermionMatrix
    //Example of BF
    /*NSL::TimeTensor<c10::complex<double>> phi({2,2});
    NSL::Tensor<c10::complex<double>> expKappa({2,2});
    (NSL::TestingExpDisc::BF(phi, expKappa)).print();
    std::cout<<phi.data_<<std::endl;
    std::cout<<expKappa.data_<<std::endl;*/
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
    return 0;
} //end of int main
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
//Benchmarking
//Tensor class
    //Vector Construction
/*    static void BM_Vector_construction(benchmark::State& state) {
        for (auto _ : state) {
            NSL::Tensor<float> example_tensor20(2);
        }
    }
    // Register the function as a benchmark
    BENCHMARK(BM_Vector_construction);

    //Tensor Construction
    static void BM_Tensor_construction(benchmark::State& state) {
        for (auto _ : state) {
            NSL::Tensor<float> example_tensor20({2,2,2});
        }
    }
    BENCHMARK(BM_Tensor_construction);

    //TimeVector Construction
    static void BM_TimeVector_construction(benchmark::State& state) {
        for (auto _ : state) {
            NSL::TimeTensor<float> example_tensor20(2);
        }
    }
    BENCHMARK(BM_TimeVector_construction);

    //TimeTensor Construction
    static void BM_TimeTensor_construction(benchmark::State& state) {
        for (auto _ : state) {
            NSL::TimeTensor<float> example_tensor20({2,2,2});
        }
    }
    BENCHMARK(BM_TimeTensor_construction);*/

/*
    static void BM_fermionMatrix(benchmark::State & state){
        NSL::TimeTensor<c10::complex<double>> phi({2,2});
        NSL::Tensor<c10::complex<double>> expKappa({2,2});
        for (auto _ : state){
            NSL::TestingExpDisc::BF(phi, expKappa);
        }
    }
    BENCHMARK(BM_fermionMatrix);
*/

/*static void BM_fermionMatrix(benchmark::State & state){
    NSL::TimeTensor<c10::complex<double>> phi({2,2});
    for (auto _ : state){
        NSL::TimeTensor<c10::complex<double>> ex(phi);
    }*/
//}
//BENCHMARK(BM_fermionMatrix);

    //Main
//    BENCHMARK_MAIN();
