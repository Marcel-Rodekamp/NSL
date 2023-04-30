#include "../test.hpp"
#include <type_traits>

//! \file Tests/Tensor/test_operatorAddition.cpp
/*!
 * To test the (elementwise) addition of a `NSL::Tensor` we need to check
 * that expected addition behaviour is satisfied as well as the data locality
 * is maintained.
 * Therefore, we demand:
 * - The +/- operations on the Tensors must match (elementwise) the +/- operations of the underlying data type.
 * - Subtracting the same tensor from itself must give 0
 * - Identity element: `NSL::Tensor = 0`
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
 * - operator+
 * - operator-
 * - operator+=
 * - operator-=
 *
 * */

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void addition(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void additionEqual(SizeTypes ... sizes);

NSL_TEST_CASE("Tensor 1D Floating Point Addition", "[Tensor,1D,Floating Point Addition]"){
    // addition is not defined for bool tensors
    if constexpr (!std::is_same<TestType,bool>::value ){
        NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
        addition<TestType>(size0);
        additionEqual<TestType>(size0);
    } 
}

NSL_TEST_CASE("Tensor 2D Floating Point Addition", "[Tensor,2D,Floating Point Addition]"){
    // addition is not defined for bool tensors
    if constexpr (!std::is_same<TestType,bool>::value ){
        NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
        NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
        addition<TestType>(size0,size1);
        additionEqual<TestType>(size0,size1);
    } 
}

NSL_TEST_CASE("Tensor 3D Floating Point Addition", "[Tensor,3D,Floating Point Addition]"){
    // addition is not defined for bool tensors
    if constexpr (!std::is_same<TestType,bool>::value ){
        NSL::size_t size0 = GENERATE(1,2,4,8,16);
        NSL::size_t size1 = GENERATE(1,2,4,8,16);
        NSL::size_t size2 = GENERATE(1,2,4,8,16);
        addition<TestType>(size0,size1,size2);
        additionEqual<TestType>(size0,size1,size2);
    } 
}

NSL_TEST_CASE("Tensor 4D Floating Point Addition", "[Tensor,4D,Floating Point Addition]"){
    // addition is not defined for bool tensors
    if constexpr (!std::is_same<TestType,bool>::value ){
        NSL::size_t size0 = GENERATE(1,2,4,8);
        NSL::size_t size1 = GENERATE(1,2,4,8);
        NSL::size_t size2 = GENERATE(1,2,4,8);
        NSL::size_t size3 = GENERATE(1,2,4,8);
        addition<TestType>(size0,size1,size2,size3);
        additionEqual<TestType>(size0,size1,size2,size3);
    } 
}


//=======================================================================
// Implementation Details: operator+, operator-
//=======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void addition(SizeTypes ... sizes){
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

    // check that the addresses are distinct (operator+ is not inplace)
    addrCheck(A,B,C,D);

    C = A + B;
    D = A - B;

    // check that the addresses are distinct (operator+ is not inplace)
    addrCheck(A,B,C,D);

    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( C[i] == A[i]+B[i] );
        REQUIRE( D[i] == A[i]-B[i] );
    }

    C = A + static_cast<Type>(1);
    D = A - static_cast<Type>(1);

    // check that the addresses are distinct (operator+ is not inplace)
    addrCheck(A,B,C,D);

    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( C[i] == A[i]+static_cast<Type>(1) );
        REQUIRE( D[i] == A[i]-static_cast<Type>(1) );
    }


    C = A - A;
    D = A + (-1*A);
    
    // Check that the addresses are still distinct (operator+ is not inplace)
    addrCheck(A,B,C,D);
    
    // check that same subtract to zero
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( C[i] == static_cast<Type>(0) );
        REQUIRE( D[i] == static_cast<Type>(0) );
    }


    C = A + 0;
    D = 0 + A;

    // Check that the addresses are still distinct (operator+ is not inplace)
    addrCheck(A,B,C,D);

    // check that the scalar 0 add workes
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( C[i] == A[i] );
        REQUIRE( D[i] == A[i] );
    }

    
    B = 0; // B is now zero everywhere
    C = A + B;
    D = B + A;

    // Check that the addresses are still distinct (operator+ is not inplace)
    addrCheck(A,B,C,D);

    // check that the scalar 0 add workes
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( B[i] == static_cast<Type>(0) );
        REQUIRE( C[i] == A[i] );
        REQUIRE( D[i] == A[i] );
    }


    // The same thing again but with subtraction
    C = A - 0;
    D = 0 - A;

    // Check that the addresses are still distinct (operator+ is not inplace)
    addrCheck(A,B,C,D);

    // check that the scalar 0 add workes
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( C[i] == A[i] );
        REQUIRE( D[i] == -A[i] );
    }

    
    B = 0; // B is now zero everywhere
    C = A - B;
    D = B - A;

    // Check that the addresses are still distinct (operator+ is not inplace)
    addrCheck(A,B,C,D);

    // check that the tensor 0 add workes
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( B[i] == static_cast<Type>(0) );
        REQUIRE( C[i] == A[i] );
        REQUIRE( D[i] == -A[i] );
    }
}


//=======================================================================
// Implementation Details: operator+=, operator-=
//=======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void additionEqual(SizeTypes ... sizes){
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


    A+=B;
    C-=D;
    // check that the addresses are distinct and maintained (operator+= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);
    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( A[i] == Abak[i]+Bbak[i] );
        REQUIRE( C[i] == Cbak[i]-Dbak[i] );
    }
    // reset the data
    A = Abak;
    B = Bbak;
    C = Cbak;
    D = Dbak;
    // check that the addresses are distinct and maintained (operator+= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);


    A+=static_cast<Type>(1);
    B-=static_cast<Type>(1);
    // check that the addresses are distinct and maintained (operator+= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);
    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( A[i] == Abak[i]+static_cast<Type>(1) );
        REQUIRE( B[i] == Bbak[i]-static_cast<Type>(1) );
    }
    // reset the data
    A = Abak;
    B = Bbak;
    // check that the addresses are distinct and maintained (operator+= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);


    A+=(-1)*A;
    B-=B;
    // check that the addresses are distinct and maintained (operator+= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);
    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( A[i] == static_cast<Type>(0) );
        REQUIRE( B[i] == static_cast<Type>(0) );
    }
    // reset the data
    A = Abak;
    B = Bbak;
    // check that the addresses are distinct and maintained (operator+= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);


    A+=0;
    B-=0;
    // check that the addresses are distinct and maintained (operator+= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);
    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( A[i] == Abak[i] );
        REQUIRE( B[i] == Bbak[i] );
    }

    C = 0;
    A+=C;
    B-=C;
    // check that the addresses are distinct and maintained (operator+= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);
    // check elementwise agree ment
    for(NSL::size_t i = 0; i < numElements; ++i){
        REQUIRE( C[i] == static_cast<Type>(0) );
        REQUIRE( A[i] == Abak[i] );
        REQUIRE( B[i] == Bbak[i] );
    }

    // reset the data
    A = Abak;
    B = Bbak;
    C = Cbak;
    // check that the addresses are distinct and maintained (operator+= is inplace)
    addrCheckDist(A,B,C,D);
    addrCheckMaint(A,B,C,D);
}
