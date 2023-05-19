#include "../test.hpp"
#include "NSL.hpp"

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_adjoint(SizeTypes ... Ns);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_H(SizeTypes ... Ns);

FLOAT_NSL_TEST_CASE("Tensor 1D Adjoint", "[Tensor,1D,Adjoint]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    test_adjoint<TestType>(size0);
}

FLOAT_NSL_TEST_CASE("Tensor 2D Adjoint", "[Tensor,2D,Adjoint]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    NSL::size_t size1 = GENERATE(1,8,32);
    test_adjoint<TestType>(size0,size1);
}

FLOAT_NSL_TEST_CASE("Tensor 3D Adjoint", "[Tensor,3D,Adjoint]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    NSL::size_t size1 = GENERATE(1,8,32);
    NSL::size_t size2 = GENERATE(1,8,32);
    test_adjoint<TestType>(size0,size1,size2);
}

FLOAT_NSL_TEST_CASE("Tensor 4D Adjoint", "[Tensor,4D,Adjoint]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    NSL::size_t size1 = GENERATE(1,8,32);
    NSL::size_t size2 = GENERATE(1,8,32);
    NSL::size_t size3 = GENERATE(1,8,32);
    test_adjoint<TestType>(size0,size1,size2,size3);
}



FLOAT_NSL_TEST_CASE("Tensor 1D H", "[Tensor,1D,H]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    test_H<TestType>(size0);
}

FLOAT_NSL_TEST_CASE("Tensor 2D H", "[Tensor,2D,H]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    NSL::size_t size1 = GENERATE(1,8,32);
    test_H<TestType>(size0,size1);
}

FLOAT_NSL_TEST_CASE("Tensor 3D H", "[Tensor,3D,H]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    NSL::size_t size1 = GENERATE(1,8,32);
    NSL::size_t size2 = GENERATE(1,8,32);
    test_H<TestType>(size0,size1,size2);
}

FLOAT_NSL_TEST_CASE("Tensor 4D H", "[Tensor,4D,H]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    NSL::size_t size1 = GENERATE(1,8,32);
    NSL::size_t size2 = GENERATE(1,8,32);
    NSL::size_t size3 = GENERATE(1,8,32);
    test_H<TestType>(size0,size1,size2,size3);
}

//=======================================================================
// Implementation Details: test_adjoint
//=======================================================================

//Test for the function adjoint (adjoint with shallow copy)
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_adjoint(SizeTypes ... Ns){

    NSL::Tensor<Type> A(Ns...);A.rand();
    Type * addr_bak = A.data();
    NSL::Tensor<Type> B(A,true);

    for(NSL::size_t d1 = 0; d1 < sizeof...(Ns); ++d1){
        for(NSL::size_t d2 = 0; d2 < sizeof...(Ns); ++d2){
			// in this case A and B change every iteration
            // .T creates a copy of A
            NSL::Tensor<Type> B = A.T(d1,d2).conj();
            
            // This adjoint mutates A by definition hence the checks 
            // that A == C and A == B
            NSL::Tensor<Type> C = A.adjoint(d1,d2);

			// check that data agrees
            REQUIRE((B == C).all());
            REQUIRE((A == C).all());
            REQUIRE((A == B).all());
            // We can't ensure that the address of C,B compares to addr_bak
            // as .adjoint changes on the memory of A.

        }
    }
}


//=======================================================================
// Implementation Details: test_H
//=======================================================================

//Test for the function H (adjoint with deepcopy)
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_H(SizeTypes ... Ns){

    NSL::Tensor<Type> A(Ns...);A.rand();
    Type * addr_bak = A.data();

    for(NSL::size_t d1 = 0; d1 < sizeof...(Ns); ++d1){
        for(NSL::size_t d2 = 0; d2 < sizeof...(Ns); ++d2){
			// in this case A stays constant, so B has to be copied every iteration
    		NSL::Tensor<Type> B = A.T(d1,d2).conj();
            
            NSL::Tensor<Type> C = A.H(d1,d2);

			// check that data agrees
            REQUIRE((B == C).all());
			// check that the correct tensors share/ not share their data
            REQUIRE( A.data() == addr_bak );
            REQUIRE( C.data() != addr_bak );
            REQUIRE( B.data() != addr_bak );

        }
    }
}


