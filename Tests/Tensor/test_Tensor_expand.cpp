#include "../test.hpp"

// Torch requirement
using size_type = long int;
template<NSL::Concept::isComplex Type, NSL::Concept::isIntegral ... SizeTypes>
void test_conj_transpose(SizeTypes ... Ns);

FLOAT_NSL_TEST_CASE("Tensor 1D Expand", "[Tensor,1D,Expand]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    test_Tensor_expand<TestType>(size0);
}

FLOAT_NSL_TEST_CASE("Tensor 2D Expand", "[Tensor,2D,Expand]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    NSL::size_t size1 = GENERATE(1,8,32);
    test_Tensor_expand<TestType>(size0,size1);
}

FLOAT_NSL_TEST_CASE("Tensor 3D Expand", "[Tensor,3D,Expand]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    NSL::size_t size1 = GENERATE(1,8,32);
    NSL::size_t size2 = GENERATE(1,8,32);
    test_Tensor_expand<TestType>(size0,size1,size2);
}

FLOAT_NSL_TEST_CASE("Tensor 4D Expand", "[Tensor,4D,Expand]"){
    NSL::size_t size0 = GENERATE(1,8,32);
    NSL::size_t size1 = GENERATE(1,8,32);
    NSL::size_t size2 = GENERATE(1,8,32);
    NSL::size_t size3 = GENERATE(1,8,32);
    test_Tensor_expand<TestType>(size0,size1,size2,size3);
}
//=======================================================================
// Implementation Details: test_conj_transpose
//=======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_Tensor_expand(SizeTypes ... Ns){

    std::array<NSL::size_t, sizeof...(Ns)> size_arr = {Ns...};
    auto expand_size = size_arr[0];

    NSL::Tensor<Type> A(Ns...);A.rand();
    NSL::Tensor<Type> B(Ns...);B.rand();
    NSL::Tensor<Type> AB_expanded = (A*B).expand(expand_size);

    // NSL::Tensor<Type> AB_expanded = NSL::LinAlg::mat_mul(A, B).expand(expand_size);

    NSL::Tensor<Type> A_expanded = A.expand(expand_size);
    NSL::Tensor<Type> B_expanded = B.expand(expand_size);
    NSL::Tensor<Type> A_B_expanded= A_expanded*B_expanded;
    // NSL::Tensor<Type> A_B_expanded = NSL::LinAlg::mat_mul(A_expanded, B_expanded);

    REQUIRE( AB_expanded.dim() == A_B_expanded.dim() );
    REQUIRE( AB_expanded.shape() == A_B_expanded.shape() );
    
}
