#include "NSL.hpp"
#include <ios>

/*! 
 * This example discusses the way how the Tensor data is located. 
 * It is important to differentiate between Tensor and View. 
 * While the Tensor holds the underlying data and the associated shape information
 * a view holds a reference on the data of another Tensor with potentially 
 * different shape information. Both instances are stored as `NSL::Tensor`.
 * This semantic is useful to reduce the amount of memory copy. For example
 * one can pass a tensor to a function (by value) which creates a view of 
 * that particular tensor with the same shape information (similar to a reference).
 * On the other site slicing a tensor does not require to copy the data 
 * but can be done by creating a view with different shape information.
 *
 * The following function will go through the different methods that perform
 * a deep copy (actually copying the data creating a new tensor) or perform
 * a shallow copy (creating a view of a tensor).
 * */

void passing_a_tensor_by_value(NSL::Tensor<double> T, double * addr_compare){
    // when we pass a tensor by value a view is created thus all the 
    // elements should have the same address as the original tensor (addr_compare).
    
    std::cout << "Passing a tensor by value also has the same addresses:" << std::endl;
    for(NSL::size_t index = 0; index < T.shape(0); ++index){
        std::cout << std::boolalpha << (&T(index) == addr_compare +index) << std::endl;
    }


}

int main(){
    // Initialize a tensor of one dimension with 4 elements; 
    // i.e. a 4-D vector
    NSL::size_t N = 4;
    NSL::Tensor<double> T(N);
    
    // It is useful to know the address of every element of the Tensor. 
    std::cout << "Original tensor addresses:" << std::endl;
    for(NSL::size_t index = 0; index < N; ++index){
        std::cout << &T(index) << std::endl;
    }
    // As you see it is stored contiguesly in memory 
    //
    /*Copy Construction*/
    // By default copy constructing a Tensor creates a view of that Tensor
    // i.e. a shallow copy
    NSL::Tensor<double> T_copy(T);
    // we can check that the addresses align:
    std::cout << "Copy constructed tensor has the same addresses:" << std::endl;
    for(NSL::size_t index = 0; index < N; ++index){
        std::cout << std::boolalpha << (&T(index) == &T_copy(index)) << std::endl;
    }

    // This is also true if we assign a tensor to initialize a new tensor
    // as this calls the same function (copy constructor)
    NSL::Tensor<double> T_copy_assign = T;
    // we can check that the addresses align:
    std::cout << "Copy constructed tensor (by assignment) has the same addresses:" << std::endl;
    for(NSL::size_t index = 0; index < N; ++index){
        std::cout << std::boolalpha << (&T(index) == &T_copy_assign(index)) << std::endl;
    }

    // As mentioned before this results in the effect that we can pass 
    // tensors by value without actually copying the data
    passing_a_tensor_by_value(T,T.data());

    /*Slicing*/
    // If we slice a tensor, e.g. taking evey second element, the resulting 
    // NSL::Tensor (view) also maintains the same data
    std::cout << "Slicing a tensor uses the same memory space:" << std::endl;
    for(NSL::size_t index = 0; index < N/2; ++index){
        std::cout << std::boolalpha << (&T(2*index) == &T(NSL::Slice(0,N,2))(index)) << std::endl;
    }

    // Sometimes however, we want to make an explicit copy of the data.
    // This can be done by a special copy constructor which takes an additional
    // bool to specify if a deep (true) or shallow (false) copy is performed
    NSL::Tensor<double> T_deepcopy(T,true);
    std::cout << "Deep copy constructed tensor doesn't have the same addresses:" << std::endl;
    for(NSL::size_t index = 0; index < N; ++index){
        std::cout << std::boolalpha << (&T(index) != &T_deepcopy(index)) << std::endl;
    }

    // The assignment operator also performs a deepcopy, which is different
    // to the case where we use = to construct a tensor as this calls a 
    // different function (NSL::Tensor::operator=).
    NSL::Tensor<double> T_deepcopy_assignment(N);
    T_deepcopy_assignment = T;
    std::cout << "Assignment operator doesn't have the same addresses:" << std::endl;
    for(NSL::size_t index = 0; index < N; ++index){
        std::cout << std::boolalpha << (&T(index) != &T_deepcopy_assignment(index)) << std::endl;
    }

    return EXIT_SUCCESS;
}
