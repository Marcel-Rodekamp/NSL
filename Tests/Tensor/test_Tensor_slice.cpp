#include "../test.hpp"
#include <sstream>
#include <string>

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_slice(SizeTypes ... sizes);

FLOAT_NSL_TEST_CASE("Tensor slice 1D", "[Tensor,slice,1D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    test_slice<TestType>(size0);
}


REAL_NSL_TEST_CASE("Tensor slice 2D", "[Tensor,slice,2D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    test_slice<TestType>(size0,size1);
}

FLOAT_NSL_TEST_CASE("Tensor slice 3D", "[Tensor,slice,3D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16);
    NSL::size_t size1 = GENERATE(1,2,4,8,16);
    NSL::size_t size2 = GENERATE(1,2,4,8,16);
    test_slice<TestType>(size0,size1,size2);
}

FLOAT_NSL_TEST_CASE("Tensor slice 4D", "[Tensor,slice,4D]"){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    NSL::size_t size1 = GENERATE(1,2,4,8);
    NSL::size_t size2 = GENERATE(1,2,4,8);
    NSL::size_t size3 = GENERATE(1,2,4,8);
    test_slice<TestType>(size0,size1,size2,size3);
}

// ======================================================================
// Implementation Detail: test_slice()
// ======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void test_slice(SizeTypes ... sizes){
    std::array<NSL::size_t, sizeof...(sizes)> sizes_arr = {sizes...};
    NSL::Tensor<Type> A(sizes...); A.rand();
    const Type * A_addr = A.data();

    std::string sizes_str = "";
    for (NSL::size_t d = 0; d < sizeof...(sizes); ++d){
        sizes_str+= std::to_string(sizes_arr[d]) + " ";
    }
    INFO("sizes : " + sizes_str);

    for(NSL::size_t d = 0; d < sizeof...(sizes); ++d){
        NSL::size_t numelModd = A.numel()/A.shape(d);

        for(NSL::size_t start = 0; start < sizes_arr[d]; ++start){
        for(NSL::size_t stop = start+1; stop < sizes_arr[d]; ++stop){
        for(NSL::size_t step = 1; step < stop - start; ++step){
            
            NSL::size_t sliceSize = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop-start)/step)); 

            NSL::Tensor<Type> Aslice = A.slice(d,start,stop,step);

            INFO( "d/D  : " + std::to_string(d) + "/" + std::to_string(sizeof...(sizes)) );
            INFO( "start: " + std::to_string(start) );
            INFO( "stop : " + std::to_string(stop) );
            INFO( "step : " + std::to_string(step) );
            INFO( "size : " + std::to_string(sliceSize) );

            // check dimension
            REQUIRE( Aslice.dim() == A.dim() );
            // check shape
            REQUIRE( Aslice.shape(d) == sliceSize );
            // check number of elements
            REQUIRE( Aslice.numel() == numelModd*sliceSize );

            // check data locality
            // The slice may throw away the first element thus comparing 
            // the addresses is not as strait forward.
            // If we e.g. slice the 0th dimension, the first element of 
            // the view `Aslice(0,...)` is the same as `A(start,...)`.
            // otherwise the first element remains the same.
            // This element is the defining address refered to by 
            // NSL::Tensor::data hence we have to check:
            REQUIRE( Aslice.data() == A_addr + start*A.strides(d) );
        }}}
    }
}


