#include <complex>
#include "../test.hpp"
#include "Tensor/tensor.hpp"
#include <typeinfo>

// Torch requirement
using size_type = long int;

template<typename T, typename ...Tsize>
void test_constructor(const Tsize &... size){
    // size: number of elements
    // Initializes a d-D Tensor of type T
    // with size elements and checks if it is initialized to 0.
    // Note: Requires NSL::Tensor::operator==(Type);
    // Note: Requires conversion from int to type T

    NSL::TimeTensor<T> Tr(size...);
    REQUIRE((Tr == 0).all());
}

template<typename T>
void test_random_access_1D(const size_type & size){
    // const std::size_t & size, number of elements
    // Initializes a 1D Tensor of type T
    // with size elements and checks if it is initialized to 0.
    // Note: NSL::Tensor::data()
    // Note: Requires conversion from int to type T
    NSL::TimeTensor<T> Tr(size);

    // Step one
    // Check element wise initialization
    // Set Value to id
    for(int i = 0; i < size; ++i){
        INFO("type = " << typeid(T).name() << ", size = " << size << ", id = " << i);
        // check that the type is returned correctly
        REQUIRE(typeid(Tr(i)) == typeid(T));

        // check for correct initialization + random access
        REQUIRE(Tr(i) == T(0));
        // check setter
        Tr(i) = static_cast<T>(i);
        // check that value is set correctly
        REQUIRE(Tr(i) == static_cast<T>(i));
        // check that underlying data ptr is changed
        REQUIRE(Tr.data()[i] == Tr(i));
    }
}

template<typename T>
void test_random_access_2D(const size_type & size0,const size_type & size1){
    // const std::size_t & size, number of elements
    // Initializes a 1D Tensor of type T
    // with size elements and checks if it is initialized to 0.
    // Note: NSL::Tensor::data()
    // Note: Requires conversion from int to type T
    // Note: Assumes data being aligned as LinearIndex = i*size1 + j if Tr(i,j) is referenced
    NSL::TimeTensor<T> Tr(size0,size1);

    // Step one
    // Check element wise initialization
    // Set Value to id
    for(int i = 0; i < size0; ++i){
        for(int j = 0; j < size1; ++j) {
            INFO("type = " << typeid(T).name() << ", size0 = " << size0 << ", size1 = " << size1 << ", id = " << i);
            // check that the type is returned correctly
            REQUIRE(typeid(Tr(i, j)) == typeid(T));

            const size_type testVal = i * size1 + j;

            // check for correct initialization + random access
            REQUIRE(Tr(i,j) == T(0));
            // check setter
            Tr(i,j) = static_cast<T>(testVal);
            // check that value is set correctly
            REQUIRE(Tr(i,j) == static_cast<T>(testVal));
            // ch`eck that underlying data ptr is changed
            REQUIRE(Tr.data()[testVal] == Tr(i,j));

        }
    }
}

template<typename T>
void test_random_access_3D(const size_type & size0,const size_type & size1,const size_type & size2){
    // const std::size_t & size, number of elements
    // Initializes a 1D Tensor of type T
    // with size elements and checks if it is initialized to 0.
    // Note: NSL::Tensor::data()
    // Note: Requires conversion from int to type T
    // Note: Assumes data being aligned as LinearIndex = i*size1*size2 + j*size1 + k if Tr(i,j) is referenced
    NSL::TimeTensor<T> Tr(size0,size1,size2);

    // Step one
    // Check element wise initialization
    // Set Value to id
    for(int i = 0; i < size0; ++i){
        for(int j = 0; j < size1; ++j) {
            for(int k = 0; k<size2; ++k) {
                INFO("type = " << typeid(T).name() << ", size0 = " << size0 << ", size1 = " << size1 << ", id = " << i);
                // check that the type is returned correctly
                REQUIRE(typeid(Tr(i, j, k)) == typeid(T));

                const size_type testVal = i * size2 * size1 + j * size2 + k;

                // check for correct initialization + random access
                REQUIRE(Tr(i,j,k) == T(0));
                // check setter
                Tr(i,j,k) = static_cast<T>(testVal);
                // check that value is set correctly
                REQUIRE(Tr(i,j,k) == static_cast<T>(testVal));
                // ch`eck that underlying data ptr is changed
                REQUIRE(Tr.data()[testVal] == Tr(i,j,k));
            }
        }
    }
}



// =============================================================================
// Test Cases
// =============================================================================


// =============================================================================
// Constructors
// =============================================================================

NSL_TEST_CASE( "TIMETENSOR: 1D Constructor", "[Tensor,Constructor,1D]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);

    test_constructor<TestType>(size);
}

NSL_TEST_CASE( "TIMETENSOR: 2D Constructor", "[Tensor,Constructor,2D]" ) {
    const size_type size0 = GENERATE(1, 100, 200);
    const size_type size1 = GENERATE(1, 100, 200);

    test_constructor<TestType>(size0,size1);
}

NSL_TEST_CASE( "TIMETENSOR: 3D Constructor", "[Tensor,Constructor,3D]" ) {
    const size_type size0 = GENERATE(1, 100, 200);
    const size_type size1 = GENERATE(1, 100, 200);
    const size_type size2 = GENERATE(1, 100, 200);

    test_constructor<TestType>(size0,size1,size2);
}


// =============================================================================
// Random Access
// =============================================================================

NSL_TEST_CASE( "TIMETENSOR: 1D Random access", "[Tensor,Random Access, 1D"){
   const size_type size = GENERATE(1, 100, 200);

    test_random_access_1D<TestType>(size);
}

NSL_TEST_CASE( "TIMETENSOR: 2D Random access", "[Tensor,Random Access, 2D"){
    const size_type size0 = GENERATE(1, 100, 200);
    const size_type size1 = GENERATE(1, 100, 200);

    test_random_access_2D<TestType>(size0,size1);
}

NSL_TEST_CASE( "TIMETENSOR: 3D Random access", "[Tensor,Random Access, 3D"){
    const size_type size0 = GENERATE(1, 10, 20);
    const size_type size1 = GENERATE(1, 10, 20);
    const size_type size2 = GENERATE(1, 10, 20);

    test_random_access_3D<TestType>(size0,size1,size2);
}
