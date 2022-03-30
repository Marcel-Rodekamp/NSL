#include "../test.hpp"
#include "NSL.hpp"

#include <typeinfo>
#include <string>

//! \file Tests/Tensor/test_constructor.cpp
/*!
 * To test constructors, we need to ensure that any members are initialized.
 * Therefore, we can demand:
 * - The data is initialized, i.e. lays in memory and is assigned a value.
 *      - The constructor defines the layout of the data. (to be tested)
 * - The member functions are accessible.
 *      - This is automatic as long as the construcors are not illformated
 *
 * To test for this we assume that `NSL::Tensor::data` is available and 
 * returns the correct memory address of the data.
 * Furthermore, we query the layout stats requiring that `NSL::Tensor::numel`,
 * `NSL:Tensor::shape` and `NSL::Tensor::dim` are returning the correct quanties.
 * All mentioned function are (almost) trivially passed to the underlying library.
 *
 * Providing Tests for:
 * - default constructor
 * - D-dimensional constructor
 * - copy constructor
 * - move constructor
 **/

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void dimConstructor(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void copyConstructor(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void moveConstructor(SizeTypes ... sizes);


NSL_TEST_CASE("Tensor Default Constructor", "[Tensor,Default,Constructor]"){
    INFO("type: " + std::string(typeid(TestType).name()));

    // default constructor
    // Creates a none element Tensor.
    // The underlying torch::Tensor carries the property "not defined"
    NSL::Tensor<TestType> T;

    REQUIRE(!torch::Tensor(T).defined());
}

NSL_TEST_CASE("Tensor 1D Constructor", "[Tensor,1D,Constructor]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    dimConstructor<TestType>(size0);
    copyConstructor<TestType>(size0);
    moveConstructor<TestType>(size0);

}

NSL_TEST_CASE("Tensor 2D Constructor", "[Tensor,2D,Constructor]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    dimConstructor<TestType>(size0,size1);
    copyConstructor<TestType>(size0,size1);
    moveConstructor<TestType>(size0,size1);

}

NSL_TEST_CASE("Tensor 3D Constructor", "[Tensor,3D,Constructor]"){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    NSL::size_t size1 = GENERATE(1,2,4,8);
    NSL::size_t size2 = GENERATE(1,2,4,8);
    dimConstructor<TestType>(size0,size1,size2);
    copyConstructor<TestType>(size0,size1,size2);
    moveConstructor<TestType>(size0,size1,size2);

}

NSL_TEST_CASE("Tensor 4D Constructor", "[Tensor,4D,Constructor]"){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    NSL::size_t size1 = GENERATE(1,2,4,8);
    NSL::size_t size2 = GENERATE(1,2,4,8);
    NSL::size_t size3 = GENERATE(1,2,4,8);
    dimConstructor<TestType>(size0,size1,size2,size3);
    copyConstructor<TestType>(size0,size1,size2,size3);
    moveConstructor<TestType>(size0,size1,size2,size3);

}


//=======================================================================
// Implementation Details: D-dimensional Constructor
//=======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void dimConstructor(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    // D-dimensional constructor
    // Constructs a tensor with dimension sizeof...(sizes) filled with zeros
    NSL::Tensor<Type> T(sizes...);
    Type * T_ptr = T.data();

    // check that the reported dimension matches the expected
    REQUIRE(T.dim() == dim);

    // check that the reported number of elements matches the expected
    REQUIRE(T.numel() == numElements);

    // check that the reported shape matches the expected
    for(NSL::size_t d = 0; d<sizeof...(sizes); ++d){
        REQUIRE(T.shape(d) == shape[d]);
    } 

    // check that all elements in the tensor are initilized to zero
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE(T_ptr[i] == static_cast<Type>(0));
    }
}


//=======================================================================
// Implementation Details: Copy Constructor 
//=======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void copyConstructor(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    // D-dimensional constructor
    // Constructs a tensor with dimension sizeof...(sizes) filled with zeros
    // ASSUMPTION: This one works correctly 
    NSL::Tensor<Type> T(sizes...);
    Type * T_ptr = T.data();

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
// Implementation Details: Move Constructor 
//=======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void moveConstructor(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    // D-dimensional constructor--output-on-failure
    // Constructs a tensor with dimension sizeof...(sizes) filled with zeros
    // ASSUMPTION: This one works correctly 
    NSL::Tensor<Type> T(sizes...);
    Type * T_ptr = T.data();

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

