#include <iostream>
//hpp files from the project
#include "Tensor/tensor.hpp"
#include "LinAlg/mat_vec.hpp"
#include "FermionMatrix/fermionMatrix.hpp"
#include<torch/torch.h> //Pytorch
#include <vector> //Vector
#include<ctime> //To compute time
#include <benchmark/benchmark.h> //Benchmark
#include <time.h>
int main() {
//Examples for tensor-TimeTensor class.
    //Construction.
/*
    NSL::Tensor<c10::complex<float>> example_tensor0(2); //Vector
    NSL::Tensor<c10::complex<float>> example_tensor1({2,2,2}); //Tensor
    NSL::TimeTensor<c10::complex<double>> example_timetensor0 (2); //Time-vector
    NSL::TimeTensor<c10::complex<double>> example_timetensor1 ({2,2,2}); //Time-tensor
    //Copy.
    NSL::Tensor<c10::complex<float>> example_tensor2 (2); //Initialize tensor
    NSL::Tensor<c10::complex<float>> example_tensor3(example_tensor2); //Copy of tensor
    NSL::TimeTensor<c10::complex<float>> example_timetensor2(2); //Initialize Time-tensor
//    NSL::TimeTensor<c10::complex<float>> example_timetensor3(example_tensor2); //Copy of Time-tensor
*/

    //Random
   /* NSL::Tensor<float> example_tensor0({2,3});
    example_tensor0.rand().print();
    example_tensor0.rand().print();
    */

    //SHAPE.
    /*NSL::Tensor<c10::complex<float>> example_tensor4 ({2,2,2}); //Initialize tensor
    std::cout<<example_tensor4.shape(1)<<std::endl; //Print shape of input position in terminal
    NSL::TimeTensor<std::complex<float>> example_timetensor4({60,60,12}); //Initialize time-tensor
    std::cout<<example_tensortime4.shape(2)<<std::endl; //Print shape of input position in terminal
     */

    //random access.


    //Example of the use of operator *
    /*NSL::Tensor<double> example_tensor6({2,2});
    (example_tensor6[1]*3)
     NSL::TimeTensor<double> example_timetensor6({2,2});
    (example_timetensor6*2);*/

    //Example of the use of exponential
   /* NSL::Tensor<float> example_tensor7 ({20000,20000}); //Initialize tensor
    example_tensor7.exp();*/


    //Example of expand
    /*NSL::Tensor<std::complex<float>> example_tensor8({3, 2});
    std::deque<long int> dimension8 ={3, 2, 2};
    example_tensor8.expand(dimension8))
    NSL::TimeTensor<std::complex<float>> example_timetensor8({3, 2});
    example_timetensor8.expand(dimension8));*/

    //Tensor shift
  /*  NSL::TimeTensor<double> example_tensor9 ({2,2});
    example_tensor9.rand().print();
    example_tensor9.shift(1, 2).print();
*/

//===========================================================================
//Example of mat_vec.hpp. Must change data_ from private //
    //Example of mat_vec Tensor x TimeTensor
  /*  NSL::Tensor<c10::complex<float>> example_tensor20 ({2,2});
    NSL::TimeTensor<c10::complex<float>> example_timetensor20 ({2,2});
    NSL::LinAlg::mat_vec(example_tensor20, example_timetensor20).print(); //Tensor x TimeTensor
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

/*NSL::Tensor<c10::complex<double>> expo({1000, 1000});
NSL::TimeTensor<c10::complex<double>> phi({50,1000});
NSL::TimeTensor<c10::complex<double>> psi({50});

expo.rand();
phi.rand();
psi.rand();
const clock_t begin_time = clock();
auto out = NSL::TestingExpDisc::BF(phi, expo);
std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC<<std::endl;

const clock_t begin_time2 = clock();
NSL::LinAlg::foreach_timeslice(out,psi);
std::cout << float( clock () - begin_time2) /  CLOCKS_PER_SEC<<std::endl;
return 0;*/


NSL::Tensor<float> example1({2,2});
example1+=1;
NSL::Tensor<float> example2({2});
NSL::LinAlg::mat_vec(example1, example2).print();
example1.print();

} //end of int main
//===============================================================================
//===============================================================================
