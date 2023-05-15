#include "../test.hpp"

//! \file Tests/Tensor/test_Tensor_rnd.cpp
/*!
 * */

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void testRND(SizeTypes ... sizes);


FLOAT_NSL_TEST_CASE("Tensor 1D RND check", "[Tensor,1D,RND]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    testRNDfloat<TestType>(size0);
}

INTEGER_NSL_TEST_CASE("Tensor 1D RND check", "[Tensor,1D,RND]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    testRNDint<TestType>(size0);
}


//=======================================================================
// Implementation Details: Index Access
//=======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void testRNDfloat(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    // create a Tensor filled with random numbers
    NSL::Tensor<Type> T(sizes...);
    T.rand();

    // check linear index access
    for(NSL::size_t i = 0; i < numElements; ++i){
        // check value match
	std::cout<<T[i]<<std::endl;
        // REQUIRE(T[i] == static_cast<Type>(i));
    }
    
    T.randn();

    // check linear index access
    for(NSL::size_t i = 0; i < numElements; ++i){
        // check value match
	std::cout<<T[i]<<std::endl;
        // REQUIRE(T[i] == static_cast<Type>(i));
    }

    std::cout<<"Test done."<<std::endl<<std::endl;
}

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void testRNDint(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    // create a Tensor filled with random numbers
    NSL::Tensor<Type> T(sizes...);
    T.randint(25,50);

    // check linear index access
    for(NSL::size_t i = 0; i < numElements; ++i){
        // check value match
	std::cout<<T[i]<<std::endl;
        // REQUIRE(T[i] == static_cast<Type>(i));
    }

    T.randint(25);
	
    // check linear index access
    for(NSL::size_t i = 0; i < numElements; ++i){
        // check value match
	std::cout<<T[i]<<std::endl;
        // REQUIRE(T[i] == static_cast<Type>(i));
    }

    std::cout<<"Test done."<<std::endl<<std::endl;
}
