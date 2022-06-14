
#include "../test.hpp"

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void shiftTensor(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void shiftTensorBySize(SizeTypes ... sizes);


NSL_TEST_CASE("Tensor 1D Random Access", "[Tensor,1D,Random Access]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    shiftTensor<TestType>(size0);
} 

FLOAT_NSL_TEST_CASE("Tensor 1D LinAlg Shift", "[Tensor,1D,Equality]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    shiftTensorBySize<TestType>(size0);
}

FLOAT_NSL_TEST_CASE("Tensor 2D LinAlg Shift", "[Tensor,2D,Equality]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    shiftTensorBySize<TestType>(size0,size1);
}
FLOAT_NSL_TEST_CASE("Tensor 3D LinAlg Shift", "[Tensor,3D,Equality]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size2 = GENERATE(1,2,4,8,16,32);
    shiftTensorBySize<TestType>(size0,size1,size2);
}
FLOAT_NSL_TEST_CASE("Tensor 4D LinAlg Shift", "[Tensor,4D,Equality]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size2 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size3 = GENERATE(1,2,4,8,16,32);
    shiftTensorBySize<TestType>(size0,size1,size2,size3);
}

// ======================================================================
// Implementation details: shiftTensor
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void shiftTensor(SizeTypes ... sizes){
    //create new Tensor
    NSL::Tensor<Type> A(sizes...);

    // shift the tensor
    NSL::Tensor<Type> Ashift = A.shift(2);

    // check that the memory spaces are same
    REQUIRE(A.data() == Ashift.data() );
}

// ======================================================================
// Implementation details: shiftTensorBySize
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void shiftTensorBySize(SizeTypes ... sizes){

    std::array<NSL::size_t, sizeof...(sizes)> size_arr = {sizes...};
        //create new Tensor
        NSL::Tensor<Type> A(sizes...);

        //Random elements
        A.rand(); 

        //shifting the above tensor in dim 0 by the size of dimension 0 such that the shift does not affect the tensor
        NSL::Tensor<Type> A_shift = A.shift(size_arr[0]);
        
        // check for equality
        REQUIRE((A == A_shift).all() );  

}