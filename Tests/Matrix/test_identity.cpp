#include "../test.hpp"
#include "Matrix/identity.hpp"

// Torch requirement
using size_type = long int;

template <typename Type>
void test_identity(const size_type & size){
    auto i = NSL::Matrix::Identity<Type>(size);

    NSL::Tensor<Type> manual(size, size);
    for(int n=0; n<size; n++){
        manual(n,n) = 1;
    }

    REQUIRE( (manual == i).all() );

}


// =============================================================================
// Test Cases
// =============================================================================

FLOAT_NSL_TEST_CASE( "Matrix: Identity", "[Matrix,identity]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);

    test_identity<TestType>(size);
}
