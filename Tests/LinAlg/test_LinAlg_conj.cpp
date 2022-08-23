#include "../test.hpp"

// Torch requirement
using size_type = long int;

template<typename T>
void test_conj(const size_type & size){
    NSL::Tensor<T> t(size, size);
    t.rand();

    NSL::Tensor<T> conjugated = NSL::LinAlg::conj(t);

    if constexpr(!NSL::is_complex<T>()){
        REQUIRE( ( conjugated == t ).all() );
    }
    REQUIRE( ( conjugated == t.conj() ).all() );

}

// =============================================================================
// Test Cases
// =============================================================================

FLOAT_NSL_TEST_CASE( "LinAlg: conj", "[LinAlg,conj]" ) {
    const size_type size = GENERATE(1, 100, 200, 500, 1000);
    test_conj<TestType>(size);
}

/*
 * note: it'd be nice to check this for int also, but t.rand() cannot fill Tensor<int>s.
 **/
