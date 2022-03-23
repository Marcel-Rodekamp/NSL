/*!
 * this tests the different versions of `NSL::LinAlg::shift`
 * \todo Not fully developed!
 * */

#include "../test.hpp"

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void shiftTensor(SizeTypes ... sizes);

NSL_TEST_CASE("Tensor 1D Random Access", "[Tensor,1D,Random Access]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    shiftTensor<TestType>(size0);
}


// ======================================================================
// Implementation details: shiftTensor
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void shiftTensor(SizeTypes ... sizes){
    //create new Tensor
    NSL::Tensor<Type> A(sizes...);

    // shift the tensor
    NSL::Tensor<Type> Ashift = NSL::LinAlg::shift(A,1);

    // check that the memory spaces are disjunct
    REQUIRE(A.data() != Ashift.data() );
}

