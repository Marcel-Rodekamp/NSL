#include "../test.hpp"
#include <type_traits>

//! \file Tests/Tensor/test_constructor.cpp
/*!
 * To test the assignement, we need to ensure that any elements are deep copied.
 * And in case of different types are converted to the desired one.
 * Therefore, we can demand:
 * - operator= copies the data (elementwise)
 * - operator= copies scalars into the tensor (elementwise)
 * - operator= maintains memory addresses
 * - operator= can handle external `torch::Tensors`
 *
 * To test for this we assume that `NSL::Tensor::data` is available and 
 * returns the correct memory address of the data. Further, the factories
 * `NSL::Tensor::rand` and `NSL::Tensor::randint` are assumed to work correctly.
 *
 * Providing Tests for:
 * - operator=
 **/

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void assignment(SizeTypes ... sizes);

NSL_TEST_CASE("Tensor 1D Assignment", "[Tensor,1D,Assignment]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    assignment<TestType>(size0);
}

NSL_TEST_CASE("Tensor 2D Assignment", "[Tensor,2D,Assignment]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    assignment<TestType>(size0,size1);
}

NSL_TEST_CASE("Tensor 3D Assignment", "[Tensor,3D,Assignment]"){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    NSL::size_t size1 = GENERATE(1,2,4,8);
    NSL::size_t size2 = GENERATE(1,2,4,8);
    assignment<TestType>(size0,size1,size2);
}

NSL_TEST_CASE("Tensor $D Assignment", "[Tensor,4D,Assignment]"){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    NSL::size_t size1 = GENERATE(1,2,4,8);
    NSL::size_t size2 = GENERATE(1,2,4,8);
    NSL::size_t size3 = GENERATE(1,2,4,8);
    assignment<TestType>(size0,size1,size2,size3);
}


//=======================================================================
// Implementation Details: Assignment
//=======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void assignment(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    NSL::Tensor<Type> T(sizes...);
    NSL::Tensor<Type> Tassigned(sizes...);
    torch::Tensor Ttorch; 

    if constexpr(std::is_same<Type,bool>::value){
        T.randint(1);
        Ttorch = torch::randint(0,1,{sizes...}, torch::TensorOptions().dtype<Type>());
    } else if constexpr(std::is_integral<Type>::value){
        T.randint(numElements);
        Ttorch = torch::randint(0,numElements,{sizes...}, torch::TensorOptions().dtype<Type>());
    } else {
        T.rand();
        Ttorch = torch::rand({sizes...}, torch::TensorOptions().dtype<Type>() );
    }

    REQUIRE(Tassigned.data() != T.data());

    Tassigned = T;
    // check data locality is different (it should be a deep copy!)
    REQUIRE(Tassigned.data() != T.data());
    // check that the data is copied correctly
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE(Tassigned.data()[i] == T.data()[i]);
    }

    Tassigned = 0;
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE(Tassigned.data()[i] == static_cast<Type>(0));
    }

    Tassigned = Ttorch;
    // check data locality is different (it should be a deep copy!)
    REQUIRE(Tassigned.data() != Ttorch.data_ptr<Type>());
    // check that the data is copied correctly
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE(Tassigned.data()[i] == Ttorch.data_ptr<Type>()[i]);
    }

}
