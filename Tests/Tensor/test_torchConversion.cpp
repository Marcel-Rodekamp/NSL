#include "../test.hpp"
#include "NSL.hpp"

#include <typeinfo>
#include <string>

//! \file Tests/Tensor/test_torchConversion.cpp
/*!
 * To test conversion of NSL::Tensor to torch::Tensor, we need to ensure 
 * that the data agrees and that the data locality does not change.
 * Therefore, we can demand:
 * - `torch::Tensor` and `NSL::Tensor` are explicitly convertible
 *     - Their data matches
 *     - Their data location matches
 *     - Their shape, dimension and number of elements match
  * - `torch::Tensor` and `NSL::Tensor` are implicitly convertible
 *     - Their data matches
 *     - Their data location matches
 *     - Their shape, dimension and number of elements match
 *
 * To test for this we assume that `NSL::Tensor::data` is available and 
 * returns the correct memory address of the data.
 * Furthermore, we query the layout stats requiring that `NSL::Tensor::numel`,
 * `NSL:Tensor::shape` and `NSL::Tensor::dim` are returning the correct quanties.
 * All mentioned function are (almost) trivially passed to the underlying library.
 *
 * Providing Tests for:
 * - Copy/Move Constructors from Torch
 * - operator torch::Tensor
 **/

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void copyTorchConstructor(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void moveTorchConstructor(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type>
void torch_nsl(torch::Tensor T_torch, NSL::Tensor<Type> T_NSL);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void explicitConversion(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void explicitConversion(SizeTypes ... sizes);


NSL_TEST_CASE("Tensor 1D Torch Conversion", "[Tensor,1D,Torch Conversion]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    copyTorchConstructor<TestType>(size0);
    moveTorchConstructor<TestType>(size0);
    explicitConversion<TestType>(size0);
    implicitConversion<TestType>(size0);
}

NSL_TEST_CASE("Tensor 2D Torch Conversion", "[Tensor,2D,Torch Conversion]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    explicitConversion<TestType>(size0,size1);
    implicitConversion<TestType>(size0,size1);
    copyTorchConstructor<TestType>(size0,size1);
    moveTorchConstructor<TestType>(size0,size1);
}

NSL_TEST_CASE("Tensor 3D Torch Conversion", "[Tensor,3D,Torch Conversion]"){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    NSL::size_t size1 = GENERATE(1,2,4,8);
    NSL::size_t size2 = GENERATE(1,2,4,8);
    copyTorchConstructor<TestType>(size0,size1,size2);
    moveTorchConstructor<TestType>(size0,size1,size2);
    explicitConversion<TestType>(size0,size1,size2);
    implicitConversion<TestType>(size0,size1,size2);
}

NSL_TEST_CASE("Tensor 4D Torch Conversion", "[Tensor,4D,Torch Conversion]"){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    NSL::size_t size1 = GENERATE(1,2,4,8);
    NSL::size_t size2 = GENERATE(1,2,4,8);
    NSL::size_t size3 = GENERATE(1,2,4,8);
    copyTorchConstructor<TestType>(size0,size1,size2,size3);
    moveTorchConstructor<TestType>(size0,size1,size2,size3);
    explicitConversion<TestType>(size0,size1,size2,size3);
    implicitConversion<TestType>(size0,size1,size2,size3);
}


//=======================================================================
// Implementation Details: Copy Constructor from Torch 
//=======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void copyTorchConstructor(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    // D-dimensional constructor on the torch end
    // Constructs a tensor with dimension sizeof...(sizes) filled with zeros
    // ASSUMPTION: This one works correctly 
    torch::Tensor T = torch::zeros({sizes...}, torch::TensorOptions().dtype<Type>());
    Type * T_ptr = T.data_ptr<Type>();

    // set some values to the tensor
    for(NSL::size_t i = 0; i < numElements; ++i){
        // the static cast could also be done implicitly, I am just a little
        // more careful for the tests
        T_ptr[i] = static_cast<Type>(i);
    }

    // call the copy constructor
    NSL::Tensor<Type> Tcopy(T);
    Type * Tcopy_ptr = Tcopy.data();

    // check that the copy is a shallow copy
    REQUIRE(T_ptr == Tcopy_ptr);

    // check that the reported dimension matches the expected
    REQUIRE(Tcopy.dim() == dim);

    // check that the reported number of elements matches the expected
    REQUIRE(Tcopy.numel() == numElements);

    // check that the reported shape matches the expected
    for(NSL::size_t d = 0; d<sizeof...(sizes); ++d){
        REQUIRE(Tcopy.shape(d) == shape[d]);
    } 

    // check that all elements of the tensors agree:
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE(Tcopy_ptr[i] == T_ptr[i]);
    }
}


//=======================================================================
// Implementation Details: Move Constructor from Torch
//=======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void moveTorchConstructor(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    // D-dimensional constructor on the torch end
    // Constructs a tensor with dimension sizeof...(sizes) filled with zeros
    // ASSUMPTION: This one works correctly 
    torch::Tensor T = torch::zeros({sizes...}, torch::TensorOptions().dtype<Type>());
    Type * T_ptr = T.data_ptr<Type>();

    // set some values to the tensor
    for(NSL::size_t i = 0; i < numElements; ++i){
        // the static cast could also be done implicitly, I am just a little
        // more careful for the tests
        T_ptr[i] = static_cast<Type>(i);
    }

    // call the move constructor
    NSL::Tensor<Type> Tmove(std::move(T));
    Type * Tmove_ptr = Tmove.data();

    // check that the move did not change the data location
    // Though the data isn't accessible from T the location of the data in
    // memory should no change (it is not a deep copy)
    REQUIRE(Tmove_ptr == T_ptr);

    // check that the reported dimension matches the expected
    REQUIRE(Tmove.dim() == dim);

    // check that the reported number of elements matches the expected
    REQUIRE(Tmove.numel() == numElements);

    // check that the reported shape matches the expected
    for(NSL::size_t d = 0; d<sizeof...(sizes); ++d){
        REQUIRE(Tmove.shape(d) == shape[d]);
    } 

    // check that all elements of the tensors agree:
    for(NSL::size_t i = 0; i < numElements; ++i){
        // the static cast could also be done implicitly, I am just a little
        // more careful for the tests
        REQUIRE(Tmove_ptr[i] == static_cast<Type>(i));
    }

    // check that the original Tensor "has been moved"
    // i.e. no method nor data is accessible
    REQUIRE_THROWS(T.data());
}


//=======================================================================
// Implementation Details: Torch <--> NSL Conversion
//=======================================================================

template<NSL::Concept::isNumber Type>
void torch_nsl(torch::Tensor T_torch, NSL::Tensor<Type> T_NSL){
    // Either one of the tensors T_torch or T_NSL is converted from the others.
    
    // check that their dimension matches
    REQUIRE(T_torch.dim() == T_NSL.dim());
    // check that their number of elements matches
    REQUIRE(T_torch.numel() == T_NSL.numel());
    // check that their shape matches 
    for(NSL::size_t d = 0; d < T_NSL.dim(); ++d){
        REQUIRE(T_torch.size(d) == T_NSL.shape(d));
    }
    
    // extract C pointer
    Type * torch_ptr = T_torch.data_ptr<Type>();
    Type * NSL_ptr   = T_NSL.data();

    // check that their data matches
    for(NSL::size_t i = 0; i < T_NSL.numel(); ++i){
        REQUIRE(torch_ptr[i] == NSL_ptr[i]);
    }

    // check that their data location matches
    REQUIRE(torch_ptr == NSL_ptr);
}


//=======================================================================
// Implementation Details: Explicit Conversion
//=======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void explicitConversion(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);
    
    // create NSL::Tensor filled with zeros 
    NSL::Tensor<Type> T_NSL(sizes...);
    Type * NSL_ptr = T_NSL.data();

    // create torch::Tensor 
    torch::Tensor T_torch = torch::zeros({sizes...}, torch::TensorOptions().dtype<Type>());
    Type * torch_ptr = T_torch.data_ptr<Type>();
    
    // fill the tensors with values
    for(NSL::size_t i = 0; i < numElements; ++i){
        NSL_ptr[i] = static_cast<Type>(i);
        torch_ptr[i] = static_cast<Type>(i);
    }
    
    // explicitly convert Torch to NSL
    torch_nsl(T_torch, NSL::Tensor<Type>(T_torch));
    
    // explicitly convert NSL to Torch 
    torch_nsl(torch::Tensor(T_NSL), T_NSL);
}


//=======================================================================
// Implementation Details: Implicit Conversion
//=======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void implicitConversion(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);
    
    // create NSL::Tensor filled with zeros 
    NSL::Tensor<Type> T_NSL(sizes...);
    Type * NSL_ptr = T_NSL.data();

    // create torch::Tensor 
    torch::Tensor T_torch = torch::zeros({sizes...}, torch::TensorOptions().dtype<Type>());
    Type * torch_ptr = T_torch.data_ptr<Type>();
    
    // fill the tensors with values
    for(NSL::size_t i = 0; i < numElements; ++i){
        NSL_ptr[i] = static_cast<Type>(i);
        torch_ptr[i] = static_cast<Type>(i);
    }
    
    // implicitly convert Torch to NSL
    torch_nsl<Type>(T_torch, T_torch);
    
    // implicitly convert NSL to Torch 
    torch_nsl<Type>(T_NSL, T_NSL);
}

