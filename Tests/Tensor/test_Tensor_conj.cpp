#include "../test.hpp"
#include "NSL.hpp"

template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_conj_transpose(SizeTypes ... Ns);

FLOAT_NSL_TEST_CASE("Tensor 1D Complex Conjugate", "[Tensor,1D,Complex Conjugate]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    test_conj_transpose<TestType>(size0);
}

FLOAT_NSL_TEST_CASE("Tensor 2D Complex Conjugate", "[Tensor,2D,Complex Conjugate]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    NSL::size_t size1 = GENERATE(1,8,32);
    test_conj_transpose<TestType>(size0,size1);
}

FLOAT_NSL_TEST_CASE("Tensor 3D Complex Conjugate", "[Tensor,3D,Complex Conjugate]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    NSL::size_t size1 = GENERATE(1,8,32);
    NSL::size_t size2 = GENERATE(1,8,32);
    test_conj_transpose<TestType>(size0,size1,size2);
}

FLOAT_NSL_TEST_CASE("Tensor 4D Complex Conjugate", "[Tensor,4D,Complex Conjugate]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    NSL::size_t size1 = GENERATE(1,8,32);
    NSL::size_t size2 = GENERATE(1,8,32);
    NSL::size_t size3 = GENERATE(1,8,32);
    test_conj_transpose<TestType>(size0,size1,size2,size3);
}
//=======================================================================
// Implementation Details: test_conj_transpose
//=======================================================================
template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_conj_transpose(SizeTypes ... Ns){

    NSL::Tensor<Type> A(Ns...);A.rand();
    Type * addr_bak = A.data();
    NSL::Tensor<Type> B(A,true);

    for(NSL::size_t d1 = 0; d1 < sizeof...(Ns); ++d1){
        for(NSL::size_t d2 = 0; d2 < sizeof...(Ns); ++d2){
            B.transpose(d1,d2).conj();
            
            NSL::Tensor<Type> C = A.transpose(d1,d2).conj();

            REQUIRE((B == C).all());
            REQUIRE( A.data() == addr_bak );
            REQUIRE( C.data() == addr_bak );
            REQUIRE( B.data() != addr_bak );

        }
    }

    for(NSL::size_t d1 = 0; d1 < sizeof...(Ns); ++d1){
        for(NSL::size_t d2 = 0; d2 < sizeof...(Ns); ++d2){
            B.transpose(d1,d2).conj();
            
            NSL::Tensor<Type> C = A.conj().transpose(d1,d2);

            REQUIRE((B == C).all());
            REQUIRE( A.data() == addr_bak );
            REQUIRE( C.data() == addr_bak );
            REQUIRE( B.data() != addr_bak );

        }
    }
}


