#include "../test.hpp"
#include <numeric>

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_tensor_dim(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_tensor_shape(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_tensor_numel(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_tensor_strides(SizeTypes ... sizes);

NSL_TEST_CASE("Tensor dimension", "[Tensor,dim]"){
    test_tensor_dim<TestType>(2); 
    test_tensor_dim<TestType>(2,2); 
    test_tensor_dim<TestType>(2,2,2); 
    test_tensor_dim<TestType>(2,2,2,2); 
    test_tensor_dim<TestType>(2,2,2,2,2); 
}
NSL_TEST_CASE("Tensor shape 1D", "[Tensor,shape, 1D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,20); 
    test_tensor_shape<TestType>(size0); 
}
NSL_TEST_CASE("Tensor shape 2D", "[Tensor,shape, 2D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,20); 
    NSL::size_t size1 = GENERATE(1,2,4,8,20); 
    test_tensor_shape<TestType>(size0,size1); 
}
NSL_TEST_CASE("Tensor shape 3D", "[Tensor,shape, 3D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,20); 
    NSL::size_t size1 = GENERATE(1,2,4,8,20); 
    NSL::size_t size2 = GENERATE(1,2,4,8,20); 
    test_tensor_shape<TestType>(size0,size1,size2); 
}
NSL_TEST_CASE("Tensor shape 4D", "[Tensor,shape, 4D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,20); 
    NSL::size_t size1 = GENERATE(1,2,4,8,20); 
    NSL::size_t size2 = GENERATE(1,2,4,8,20); 
    NSL::size_t size3 = GENERATE(1,2,4,8,20); 
    test_tensor_shape<TestType>(size0,size1,size2,size3); 
}
NSL_TEST_CASE("Tensor numel 1D", "[Tensor,numel, 1D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,20); 
    test_tensor_numel<TestType>(size0); 
}
NSL_TEST_CASE("Tensor numel 2D", "[Tensor,numel, 2D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,20); 
    NSL::size_t size1 = GENERATE(1,2,4,8,20); 
    test_tensor_numel<TestType>(size0,size1); 
}
NSL_TEST_CASE("Tensor numel 3D", "[Tensor,numel, 3D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,20); 
    NSL::size_t size1 = GENERATE(1,2,4,8,20); 
    NSL::size_t size2 = GENERATE(1,2,4,8,20); 
    test_tensor_numel<TestType>(size0,size1,size2); 
}
NSL_TEST_CASE("Tensor numel 4D", "[Tensor,numel, 4D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,20); 
    NSL::size_t size1 = GENERATE(1,2,4,8,20); 
    NSL::size_t size2 = GENERATE(1,2,4,8,20); 
    NSL::size_t size3 = GENERATE(1,2,4,8,20); 
    test_tensor_numel<TestType>(size0,size1,size2,size3); 
}
NSL_TEST_CASE("Tensor strides 1D", "[Tensor,strides, 1D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,20); 
    test_tensor_strides<TestType>(size0); 
}
NSL_TEST_CASE("Tensor strides 2D", "[Tensor,strides, 2D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,20); 
    NSL::size_t size1 = GENERATE(1,2,4,8,20); 
    test_tensor_strides<TestType>(size0,size1); 
}
NSL_TEST_CASE("Tensor strides 3D", "[Tensor,strides, 3D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,20); 
    NSL::size_t size1 = GENERATE(1,2,4,8,20); 
    NSL::size_t size2 = GENERATE(1,2,4,8,20); 
    test_tensor_strides<TestType>(size0,size1,size2); 
}
NSL_TEST_CASE("Tensor strides 4D", "[Tensor,strides, 4D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,20); 
    NSL::size_t size1 = GENERATE(1,2,4,8,20); 
    NSL::size_t size2 = GENERATE(1,2,4,8,20); 
    NSL::size_t size3 = GENERATE(1,2,4,8,20); 
    test_tensor_strides<TestType>(size0,size1,size2,size3); 
}

// ======================================================================
// Implementation Details: test_tensor_dim
// ======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_tensor_dim(SizeTypes ... sizes){
    NSL::Tensor<Type> A(sizes...);

    REQUIRE(A.dim() == sizeof...(sizes));
}

// ======================================================================
// Implementation Details: test_tensor_shape
// ======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_tensor_shape(SizeTypes ... sizes){
    
    std::array<NSL::size_t, sizeof...(sizes)> sizes_arr = {sizes...};

    NSL::Tensor<Type> A(sizes...);

    // shape(d) get's the size of the d-th dimension
    for(NSL::size_t d = 0; d < sizeof...(sizes); ++d){
        REQUIRE(A.shape(d) == sizes_arr[d]);
    }
    // shape() get's the size of  the d-th dimension
    for(NSL::size_t d = 0; d < sizeof...(sizes); ++d){
        REQUIRE(A.shape()[d] == sizes_arr[d]);
    }
}

// ======================================================================
// Implementation Details: test_tensor_numel
// ======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_tensor_numel(SizeTypes ... sizes){
    
    std::array<NSL::size_t, sizeof...(sizes)> sizes_arr = {sizes...};

    NSL::size_t numel = std::accumulate(sizes_arr.begin(),sizes_arr.end(),1,std::multiplies<NSL::size_t>());

    NSL::Tensor<Type> A(sizes...);

    REQUIRE(A.numel() == numel);
}

// ======================================================================
// Implementation Details: test_tensor_strides
// ======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_tensor_strides(SizeTypes ... sizes){
    
    std::array<NSL::size_t, sizeof...(sizes)> sizes_arr = {sizes...};
    std::vector<NSL::size_t> strides_arr(sizeof...(sizes));

    strides_arr[sizeof...(sizes)-1] = 1;

    // row major strides
    // https://en.wikipedia.org/wiki/Row-_and_column-major_order#Address_calculation_in_general
    for(NSL::size_t d = sizeof...(sizes)-2; d >=0; --d){
        strides_arr[d] = strides_arr[d+1] * sizes_arr[d+1]; 
    }

    NSL::Tensor<Type> A(sizes...);

    // shape(d) get's the stride of the d-th dimension
    for(NSL::size_t d = 0; d < sizeof...(sizes); ++d){
        REQUIRE(A.strides(d) == strides_arr[d]);
    }
    // shape() get's the stride of  the d-th dimension
    for(NSL::size_t d = 0; d < sizeof...(sizes); ++d){
        REQUIRE(A.strides()[d] == strides_arr[d]);
    }

}
