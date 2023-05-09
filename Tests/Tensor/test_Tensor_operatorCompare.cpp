#include "../test.hpp"

//! \file Tests/Tensor/test_randomAccess.cpp
/*!
 * To test the comparison operators we need to check that they obey the expected
 * behaviour. 
 * Therefore, we demand
 * - The operator must match the elementwise comparison
 * - New Tensors must be created at return
 * - Errors have to be thrown if underlying data type does not provide a 
 *   sensible implementation
 *
 * For this test we assume that the following members work correctly:
 * - Factory `NSL::Tensor::rand`
 * - Factory `NSL::Tensor::randint`
 * - Linear Random Access `NSL::Tensor::operator[]`
 * - C-Pointer Access `NSL::Tensor::data`
 *
 * Providing tests for:
 * - operator==
 * - operator!=
 * - operator<
 * - operator>
 * - operator<=
 * - operator>=
 * */

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void comparison(SizeTypes ... sizes);


NSL_TEST_CASE("Tensor 1D Comparison", "[Tensor,1D,Comparison]" ){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    comparison<TestType>(size0);
}

NSL_TEST_CASE("Tensor 2D Comparison", "[Tensor,2D,Comparison]" ){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    comparison<TestType>(size0,size1);
}

NSL_TEST_CASE("Tensor 3D Comparison", "[Tensor,3D,Comparison]" ){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    NSL::size_t size1 = GENERATE(1,2,4,8);
    NSL::size_t size2 = GENERATE(1,2,4,8);
    comparison<TestType>(size0,size1,size2);
}

NSL_TEST_CASE("Tensor 4D Comparison", "[Tensor,4D,Comparison]" ){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    NSL::size_t size1 = GENERATE(1,2,4,8);
    NSL::size_t size2 = GENERATE(1,2,4,8);
    NSL::size_t size3 = GENERATE(1,2,4,8);
    comparison<TestType>(size0,size1,size2,size3);
}

// ======================================================================
// Implementation Details: comparison
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void comparison(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    // create tensors
    NSL::Tensor<Type> A(sizes...);
    NSL::Tensor<Type> B(sizes...);
    NSL::Tensor<Type> C(sizes...);
    NSL::Tensor<bool> res(sizes...);

    if constexpr(std::is_same<Type,bool>::value){
        A.randint(1);
        B.randint(1);
        C.randint(1);
    } else if constexpr(std::is_integral<Type>::value){
        A.randint(numElements);
        B.randint(numElements);
        C.randint(numElements);
    } else {
        A.rand();
        B.rand();
        C.rand();
    }

    A[0] = B[0];


    auto checkPtr = [](NSL::Tensor<Type> Al, NSL::Tensor<Type> Bl, NSL::Tensor<Type> resl){
        REQUIRE(Al.data() != Bl.data());
        REQUIRE(Al.data() != resl.data());
        REQUIRE(Bl.data() != resl.data());
    };
    // The result tensor should have the same shape/sizes
    auto checkStats = [&numElements,&dim,&shape](NSL::Tensor<Type> resl){
        REQUIRE(resl.numel() == numElements);
        REQUIRE(resl.dim() == dim);
        for(NSL::size_t d = 0; d < dim; ++d){
            REQUIRE(resl.shape(d) == shape[d]);
        }
    };

    res = (A==B);
    checkPtr(A,B,res);
    checkStats(res);
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE(res[i] == (A[i]==B[i]) );
    }

    res = (A!=B);
    checkPtr(A,B,res);
    checkStats(res);
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE(res[i] == (A[i]!=B[i]) );
    }


    if constexpr (NSL::is_complex<Type>()) {
        REQUIRE_THROWS(A>B);
        REQUIRE_THROWS(A<B);
        REQUIRE_THROWS(A>=B);
        REQUIRE_THROWS(A>=B);
    } else {

        res = (A>B);
        checkPtr(A,B,res);
        checkStats(res);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (A[i]>B[i]) );
        }
        res = (A<B);
        checkPtr(A,B,res);
        checkStats(res);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (A[i]<B[i]) );
        }
        res = (A>=B);
        checkPtr(A,B,res);
        checkStats(res);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (A[i]>=B[i]) );
        }
        res = (A<=B);
        checkPtr(A,B,res);
        checkStats(res);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (A[i]<=B[i]) );
        }


        Type compareValue = B[0];

        res = (C==compareValue);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (C[i]==compareValue) );
        }
        res = (C!=compareValue);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (C[i]!=compareValue) );
        }
        res = (C>compareValue);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (C[i]>compareValue) );
        }
        res = (C<compareValue);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (C[i]<compareValue) );
        }
        res = (C>=compareValue);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (C[i]>=compareValue) );
        }
        res = (C<=compareValue);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (C[i]<=compareValue) );
        }

        res = (compareValue==C);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (C[i]==compareValue) );
        }
        res = (compareValue!=C);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (C[i]!=compareValue) );
        }
        res = (compareValue<C);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (compareValue<C[i]) );
        }
        res = (compareValue>C);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (compareValue>C[i]) );
        }
        res = (compareValue<=C);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (compareValue<=C[i]) );
        }
        res = (compareValue>=C);
        for(NSL::size_t i = 0; i < numElements; ++i){
            REQUIRE(res[i] == (compareValue>=C[i]) );
        }

    }

}
