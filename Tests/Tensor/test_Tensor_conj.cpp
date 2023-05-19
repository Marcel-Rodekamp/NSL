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

template<NSL::Concept::isNumber Type>
void testTranspose(const NSL::Tensor<Type> & A, const NSL::Tensor<Type> & Abak, const NSL::Tensor<Type> & AH){
    for(NSL::size_t i = 0; i < AH.shape(0); ++i){
        for(NSL::size_t j = 0; j < AH.shape(1); ++j){
            REQUIRE(AH(i,j)==Abak(j,i));
            REQUIRE(AH(i,j)==A(j,i));
        } // for j
    } // for i
}

FLOAT_NSL_TEST_CASE("Tensor Transpose 3D", "[Tensor, Transpose3D]"){
    // This test scales very bad with the tensor size therefore we have to 
    // restrict it to a very small one here.
    NSL::size_t size0 = 3;//GENERATE(1,2,4);
    NSL::size_t size1 = 3;//GENERATE(1,2,4);
    NSL::size_t size2 = 3;//GENERATE(1,2,4);
   

    std::vector<NSL::size_t> sizes = {size0,size1,size2};

    NSL::Tensor<TestType> A(size0,size1,size2); A.rand();
    NSL::Tensor<TestType> Abak(A,true);

    // d1 and d2 are the dimensions which become transposed
    NSL::size_t d1, d2;

    for(NSL::size_t dn = 0; dn < 3; ++dn){

        // pick a combination of d1 and d2
        if(dn == 0){ d1=1;d2=2; }
        else if(dn == 1){ d1=0;d2=2; }
        else if(dn == 2){ d1=0;d2=1; }

        // transpose A
        NSL::Tensor<TestType> AT = A.T(d1,d2);
        for(NSL::size_t k=0; k < sizes[dn]; ++k){
            if(dn == 0){
                testTranspose(
                    A(   k,NSL::Slice(),NSL::Slice()),
                    Abak(k,NSL::Slice(),NSL::Slice()),
                    AT(  k,NSL::Slice(),NSL::Slice())
                );
            } else if(dn == 1){
                testTranspose(
                    A(   NSL::Slice(),k,NSL::Slice()),
                    Abak(NSL::Slice(),k,NSL::Slice()),
                    AT(  NSL::Slice(),k,NSL::Slice())
                );
            } else if(dn == 2){
                testTranspose(
                    A(   NSL::Slice(),NSL::Slice(),k),
                    Abak(NSL::Slice(),NSL::Slice(),k),
                    AT(  NSL::Slice(),NSL::Slice(),k)
                );
            }
        }

        // repeat with d1 and d2 interchanged
        AT = A.T(d2,d1);
        for(NSL::size_t k=0; k < sizes[dn]; ++k){
            if(dn == 0){
                testTranspose(
                    A(   k,NSL::Slice(),NSL::Slice()),
                    Abak(k,NSL::Slice(),NSL::Slice()),
                    AT(  k,NSL::Slice(),NSL::Slice())
                );
            } else if(dn == 1){
                testTranspose(
                    A(   NSL::Slice(),k,NSL::Slice()),
                    Abak(NSL::Slice(),k,NSL::Slice()),
                    AT(  NSL::Slice(),k,NSL::Slice())
                );
            } else if(dn == 2){
                testTranspose(
                    A(   NSL::Slice(),NSL::Slice(),k),
                    Abak(NSL::Slice(),NSL::Slice(),k),
                    AT(  NSL::Slice(),NSL::Slice(),k)
                );
            }
        }
    }
}

// combinatorics for a D>3 case are very hard to get right, I don't think 
// we would gain anything from those tests.
