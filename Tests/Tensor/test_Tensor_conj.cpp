#include "../test.hpp"

NSL_TEST_CASE("Tensor Transpose", "[Tensor, Transpose]"){
    NSL::size_t size = GENERATE(1,2,4,8,32,64);

    NSL::Tensor<TestType> A(size,size); 
    if constexpr(NSL::Concept::isFloatingPoint<TestType>){
        A.rand();
    } else if(NSL::Concept::isType<TestType,bool>){
        A(NSL::Slice(0,size,2)) = true;
    }  else{ 
        A.randint(0,size);
    }
    NSL::Tensor<TestType> Abak(A,true);
    
    A.transpose();

    for(NSL::size_t i = 0; i < size; ++i){
        for(NSL::size_t j = 0; j < size; ++j){
            REQUIRE(A(i,j) == Abak(j,i));
        }
    }

    // restore default
    A = Abak;

    A.transpose(0,1);

    for(NSL::size_t i = 0; i < size; ++i){
        for(NSL::size_t j = 0; j < size; ++j){
            REQUIRE(A(i,j) == Abak(j,i));
        }
    }
}

COMPLEX_NSL_TEST_CASE("Tensor Conjugate", "[Tensor, Conjugate]"){
    NSL::size_t size = GENERATE(1,2,4,8,32,64);

    NSL::Tensor<TestType> A(size,size); A.rand();
    NSL::Tensor<TestType> Abak(A,true);
    
    A.conj();

    REQUIRE( (A.imag() == -Abak.imag()).all() );
}

COMPLEX_NSL_TEST_CASE("Tensor Conjugate Transpose", "[Tensor, ConjugateTranspose]"){
    NSL::size_t size = GENERATE(1,2,4,8,32,64);

    NSL::Tensor<TestType> A(size,size); A.rand();
    NSL::Tensor<TestType> Abak(A,true);
    
    // .transpose() operates on the underlying data
    // .T() creates a new tensor
    // .conj() operates on the same memory use NSL::LinAlg::conj for creating a new Tensor
    NSL::Tensor<TestType> AH = A.T().conj();

    for(NSL::size_t i = 0; i < size; ++i){
        for(NSL::size_t j = 0; j < size; ++j){
            REQUIRE(AH(i,j) == std::conj(Abak(j,i)));
            // This test must succeed as .T first creates a new tensor
            REQUIRE(AH(i,j) == std::conj(A(j,i)));
        }
    }

    // restore default
    AH = A.conj().T();

    for(NSL::size_t i = 0; i < size; ++i){
        for(NSL::size_t j = 0; j < size; ++j){
            REQUIRE(AH(i,j) == std::conj(Abak(j,i)));
            // A.conj() conjugates A itself therefore, this test must fail
            REQUIRE_FALSE(AH(i,j) == std::conj(A(j,i)));
        }
    }

}

