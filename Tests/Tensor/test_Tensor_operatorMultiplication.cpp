#include "../test.hpp"
#include <type_traits>

//! \file Tests/Tensor/test_operatorMultiplication.cpp
/*!
 * To test the (elementwise) multiplication of a `NSL::Tensor` we need to check
 * that expected multiplication behaviour is satisfied as well as the data locality
 * is maintained.
 * Therefore, we demand:
 * - The * / / operations on the Tensors must match (elementwise) the * / / operations of the underlying data type.
 * - Deviding the same tensor from itself must give 1
 * - Identity element: `NSL::Tensor = 1`
 * - Tensor/scalar operations must be commutatively callable
 * - Data locallity must work out
 * Disclaimer: This is not a test for precision
 *
 * For this test we assume that the following members work correctly:
 * - Factory `NSL::Tensor::rand`
 * - Factory `NSL::Tensor::randint`
 * - Linear Random Access `NSL::Tensor::operator[]`
 * - C-Pointer Access `NSL::Tensor::data`
 * - Assignment `NSL::Tensor::operator=`
 *
 * Providing tests for:
 * - operator*
 * - operator/
 * - operator*=
 * - operator/=
 *
 * */

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void multiplication(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void multiplicationEqual(SizeTypes ... sizes);

FLOAT_NSL_TEST_CASE("Tensor 1D Floating Point multiplication", "[Tensor,1D,Floating Point multiplication]"){
        NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
        multiplication<TestType>(size0);
        multiplicationEqual<TestType>(size0);
}


FLOAT_NSL_TEST_CASE("Tensor 2D Floating Point multiplication", "[Tensor,2D,Floating Point multiplication]"){
        NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
        NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
        multiplication<TestType>(size0,size1);
        multiplicationEqual<TestType>(size0,size1);
}

FLOAT_NSL_TEST_CASE("Tensor 3D Floating Point multiplication", "[Tensor,3D,Floating Point multiplication]"){
        NSL::size_t size0 = GENERATE(1,2,4,8,16);
        NSL::size_t size1 = GENERATE(1,2,4,8,16);
        NSL::size_t size2 = GENERATE(1,2,4,8,16);
        multiplication<TestType>(size0,size1,size2);
        multiplicationEqual<TestType>(size0,size1,size2);
}

FLOAT_NSL_TEST_CASE("Tensor 4D Floating Point multiplication", "[Tensor,4D,Floating Point multiplication]"){
    // multiplication is not defined for bool tensors
        NSL::size_t size0 = GENERATE(1,2,4,8);
        NSL::size_t size1 = GENERATE(1,2,4,8);
        NSL::size_t size2 = GENERATE(1,2,4,8);
        NSL::size_t size3 = GENERATE(1,2,4,8);
        multiplication<TestType>(size0,size1,size2,size3);
        multiplicationEqual<TestType>(size0,size1,size2,size3);
}


//=======================================================================
// Implementation Details: operator*, operator/
//=======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void multiplication(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    // create Tensors
    NSL::Tensor<Type> A(sizes...); 
    NSL::Tensor<Type> B(sizes...); 
    NSL::Tensor<Type> C(sizes...); 
    NSL::Tensor<Type> D(sizes...); 

    if constexpr(std::is_integral<Type>::value){
        A.randint(numElements);
        B.randint(numElements);
    } else {
        A.rand();
        B.rand();
    }
    
    // ensure that no 0 is in there:
    for(NSL::size_t i = 0; i < numElements; ++i){
        if(A[i] == static_cast<Type>(0)){
            A[i] = static_cast<Type>(1);
        }
        if(B[i] == static_cast<Type>(0)){
            B[i] = static_cast<Type>(1);
        }
    }

    auto addrCheck = [](
        const NSL::Tensor<Type> &Al, 
        const NSL::Tensor<Type> &Bl, 
        const NSL::Tensor<Type> &Cl, 
        const NSL::Tensor<Type> &Dl) {
        REQUIRE(Al.data() != Bl.data()); REQUIRE(Al.data() != Cl.data()); 
        REQUIRE(Al.data() != Dl.data()); 
        REQUIRE(Bl.data() != Cl.data()); REQUIRE(Bl.data() != Dl.data()); 
        REQUIRE(Cl.data() != Dl.data()); 
    };

    // check that the addresses are distinct (operator* is not inplace)
    addrCheck(A,B,C,D);
     

    C = A * B;
    D = A / B;
    // check that the addresses are distinct (operator* is not inplace)
    addrCheck(A,B,C,D);
    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( compare_floating_point(C[i] , A[i]*B[i]) );
        REQUIRE( compare_floating_point(D[i] , A[i]/B[i]) );
    }

    C = A * static_cast<Type>(2);
    D = A / static_cast<Type>(2);
    // check that the addresses are distinct (operator* is not inplace)
    addrCheck(A,B,C,D);
    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( compare_floating_point(C[i] , A[i]*static_cast<Type>(2)) );
        REQUIRE( compare_floating_point(D[i] , A[i]/static_cast<Type>(2)) );
    }


    C = A / A;
    D = A * (1/A);
    // Check that the addresses are still distinct (operator* is not inplace)
    addrCheck(A,B,C,D);
    // check that same subtract to zero
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( compare_floating_point(C[i] , static_cast<Type>(1)) );
        REQUIRE( compare_floating_point(D[i] , static_cast<Type>(1)) );
    }


    C = A * 1;
    D = 1 * A;
    // Check that the addresses are still distinct (operator* is not inplace)
    addrCheck(A,B,C,D);
    // check that the scalar 0 add workes
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( C[i] == A[i] );
        REQUIRE( D[i] == A[i] );
    }

    
    B = 1; // B is now one everywhere
    C = A * B;
    D = B * A;
    // Check that the addresses are still distinct (operator* is not inplace)
    addrCheck(A,B,C,D);
    // check that the scalar 0 add workes
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( B[i] == static_cast<Type>(1) );
        REQUIRE( C[i] == A[i] );
        REQUIRE( D[i] == A[i] );
    }


    // The same thing again but with devision
    C = A / 1;
    D = 1 / (1/A);

    // Check that the addresses are still distinct (operator* is not inplace)
    addrCheck(A,B,C,D);

    // check that the scalar 0 add workes
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( almost_equal(C[i], A[i]) );
        REQUIRE( compare_floating_point(D[i], A[i]) );
    }

    
    B = 1; // B is now one everywhere
    C = A / B;
    D = B / (1/A);

    // Check that the addresses are still distinct (operator* is not inplace)
    addrCheck(A,B,C,D);

    // check that the tensor 0 add workes
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( B[i] == static_cast<Type>(1) );
        REQUIRE( almost_equal(C[i], A[i]) );
        REQUIRE( compare_floating_point(D[i], A[i]) );
    }
}


//=======================================================================
// Implementation Details: operator*=, operator/=
//=======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void multiplicationEqual(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    // create Tensors
    NSL::Tensor<Type> A(sizes...); 
    NSL::Tensor<Type> B(sizes...); 
    NSL::Tensor<Type> C(sizes...); 
    NSL::Tensor<Type> D(sizes...); 

    if constexpr(std::is_integral<Type>::value){
        A.randint(numElements);
        B.randint(numElements);
        C.randint(numElements);
        D.randint(numElements);
    } else {
        A.rand();
        B.rand();
        C.rand();
        D.rand();
    }

    auto addrCheckDist = [](
        const NSL::Tensor<Type> &Al, 
        const NSL::Tensor<Type> &Bl, 
        const NSL::Tensor<Type> &Cl, 
        const NSL::Tensor<Type> &Dl) {
        REQUIRE(Al.data() != Bl.data()); REQUIRE(Al.data() != Cl.data()); 
        REQUIRE(Al.data() != Dl.data()); 
        REQUIRE(Bl.data() != Cl.data()); REQUIRE(Bl.data() != Dl.data()); 
        REQUIRE(Cl.data() != Dl.data()); 
    };

    // backup the information
    NSL::Tensor<Type> Abak(sizes...);
    Abak = A;
    Type * Abak_ptr = A.data();
    NSL::Tensor<Type> Bbak(sizes...);
    Bbak = B;
    Type * Bbak_ptr = B.data();
    NSL::Tensor<Type> Cbak(sizes...);
    Cbak = C;
    Type * Cbak_ptr = C.data();
    NSL::Tensor<Type> Dbak(sizes...);
    Dbak = D;
    Type * Dbak_ptr = D.data();

    auto addrCheckMaint = [Abak_ptr,Bbak_ptr,Cbak_ptr,Dbak_ptr](
        const NSL::Tensor<Type> &Al, 
        const NSL::Tensor<Type> &Bl, 
        const NSL::Tensor<Type> &Cl, 
        const NSL::Tensor<Type> &Dl) {
        REQUIRE(Al.data() == Abak_ptr);
        REQUIRE(Bl.data() == Bbak_ptr);
        REQUIRE(Cl.data() == Cbak_ptr);
        REQUIRE(Dl.data() == Dbak_ptr);
    };


    A*=B;
    C/=D;
    // check that the addresses are distinct and maintained (operator*= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);
    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( compare_floating_point(A[i], Abak[i]*Bbak[i]) );
        REQUIRE( compare_floating_point(C[i], Cbak[i]/Dbak[i]) );
    }
    // reset the data
    A = Abak;
    B = Bbak;
    C = Cbak;
    D = Dbak;
    // check that the addresses are distinct and maintained (operator*= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);


    A*=static_cast<Type>(2);
    B/=static_cast<Type>(2);
    // check that the addresses are distinct and maintained (operator*= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);
    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( compare_floating_point(A[i], Abak[i]*static_cast<Type>(2)) );
        REQUIRE( compare_floating_point(B[i], Bbak[i]/static_cast<Type>(2)) );
    }
    // reset the data
    A = Abak;
    B = Bbak;
    // check that the addresses are distinct and maintained (operator*= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);


    A*= 1/A;
    B/=B;
    // check that the addresses are distinct and maintained (operator*= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);
    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( compare_floating_point(A[i], static_cast<Type>(1)) );
        REQUIRE( compare_floating_point(B[i], static_cast<Type>(1)) );
    }
    // reset the data
    A = Abak;
    B = Bbak;
    // check that the addresses are distinct and maintained (operator*= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);


    A*=1;
    B/=1;
    // check that the addresses are distinct and maintained (operator*= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);
    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( almost_equal(A[i], Abak[i]) );
        REQUIRE( almost_equal(B[i], Bbak[i]) );
    }

    C = 1;
    A*=C;
    B/=C;
    // check that the addresses are distinct and maintained (operator*= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);
    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( C[i] == static_cast<Type>(1) );
        REQUIRE( compare_floating_point(A[i],Abak[i]) );
        REQUIRE( compare_floating_point(B[i],Bbak[i]) );
    }

    // reset the data
    A = Abak;
    B = Bbak;
    C = Cbak;
    // check that the addresses are distinct and maintained (operator*= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);
}
