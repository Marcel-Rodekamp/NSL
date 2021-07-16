#include <iostream>
#include "Tensor/tensor.hpp"
#include <vector>

int main(){
    // test you tensors here if you like ;)

//Examples for tensor class
    //Example for tensor construction, 1D, 3D and copy. Must change data_ from private to public.
 /*  NSL::Tensor<float> example_tensor0 (12);
    std::cout<< example_tensor0.data_ << std::endl;
    NSL::Tensor<float> example_tensor1 = example_tensor0;
    std::cout<< example_tensor1.data_ << std::endl;
    std::vector<long int> dimension = {60,60,12};
    NSL::Tensor<float> example_tensor2 (dimension);
    std::cout<< example_tensor2.data_ << std::endl;*/

    //Example of the use of SHAPE.
    /*NSL::Tensor<float> example_tensor3 ({60,60,12});
    std::size_t a = 1;
    std::cout<<example_tensor3.shape(a);*/

    //Example of the use of random access.
    /*NSL::Tensor<float> example_tensor1 ({60,60,12});
    long int a = 1;
     std::cout<< example_tensor1(a,a,a) <<std::endl;*/

    //Example of the use of exponential
   /* NSL::Tensor<float> example_tensor5 ({60,60,12}); //Initialize tensor
    NSL::Tensor<float> example_tensor6 = example_tensor5.exponential();
    std::cout<< example_tensor6.data_<<std::endl;

    NSL::Tensor<float> example_tensor7 = example_tensor6.exponential();
    std::cout<< example_tensor7.data_<<std::endl;*/

//-------------------------------------------------------------------------------
// Example of Time Tensor
}
