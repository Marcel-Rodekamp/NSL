#include "../test.hpp"

NUMERIC_NSL_TEST_CASE("LinAlg: inner_product 1D", "[LinAlg,inner_product]"){
    const NSL::size_t size = GENERATE(1,100,200,500,1000);
    INFO(fmt::format("Size: {}", size));

    // define 2 tensors with ones/twos everywhere such that we can 
    // predetermine the results of the inner product
    NSL::Tensor<TestType> a(size); a = 1; 
    NSL::Tensor<TestType> b(size); b = 2;

    TestType res = NSL::LinAlg::inner_product(a,b);
    REQUIRE( res == TestType(2*size) );
    res = NSL::LinAlg::inner_product(a,a);
    REQUIRE( res == TestType(size) );
    res = NSL::LinAlg::inner_product(b,b);
    REQUIRE( res == TestType(4*size) );
}


NUMERIC_NSL_TEST_CASE("LinAlg: inner_product 2D", "[LinAlg,inner_product]"){
    const NSL::size_t size1 = GENERATE(1,100,200,500,1000);
    const NSL::size_t size2 = GENERATE(1,100,200,500,1000);
    INFO(fmt::format("Size: {}", size1));
    INFO(fmt::format("Size: {}", size2));

    // define 2 tensors with ones/twos everywhere such that we can 
    // predetermine the results of the inner product
    NSL::Tensor<TestType> a(size1,size2); a = 1; 
    NSL::Tensor<TestType> b(size1,size2); b = 2;

    TestType res = NSL::LinAlg::inner_product(a,b);
    REQUIRE( res == TestType(2*size1*size2) );
    res = NSL::LinAlg::inner_product(a,a);
    REQUIRE( res == TestType(size1*size2) );
    res = NSL::LinAlg::inner_product(b,b);
    REQUIRE( res == TestType(4*size1*size2) );
}


FLOAT_NSL_TEST_CASE("LinAlg: inner_product random", "[LinAlg,inner_product]"){
    const NSL::size_t size1 = GENERATE(1,100,200);
    const NSL::size_t size2 = GENERATE(1,100,200);
    INFO(fmt::format("Size: {}", size1));
    INFO(fmt::format("Size: {}", size2));

    // define 2 tensors randomly filled for checkup
    NSL::Tensor<TestType> a(size1,size2); a.rand(); 
    NSL::Tensor<TestType> b(size1,size2); b.rand();

    TestType ab = 0;
    TestType aa = 0;
    if constexpr( NSL::is_complex<TestType>() ){
        for(int i = 0; i < size1; ++i){
            for(int j = 0; j < size2; ++j){
                ab+= std::conj(a(i,j)) * b(i,j);
                aa+= std::conj(a(i,j)) * a(i,j);
            }
        }
    } else {
        for(int i = 0; i < size1; ++i){
            for(int j = 0; j < size2; ++j){
                ab+= a(i,j) * b(i,j);
                aa+= a(i,j) * a(i,j);
            }
        }
    }

    TestType res = NSL::LinAlg::inner_product(a,b);
    INFO(fmt::format("(a,b) = {} == {}; diff = {}",NSL::to_string(res), NSL::to_string(ab), NSL::to_string(NSL::LinAlg::abs(res-ab))));
    REQUIRE(almost_equal(res,ab,std::numeric_limits<TestType>::digits10-1));
    res = NSL::LinAlg::inner_product(a,a);
    INFO(fmt::format("(a,a) = {} == {}; diff = {}",NSL::to_string(res), NSL::to_string(aa), NSL::to_string(NSL::LinAlg::abs(res-aa))));
    REQUIRE( almost_equal(res,aa,std::numeric_limits<TestType>::digits10-1) );
    REQUIRE(NSL::real(res)>=0);
    REQUIRE( almost_equal(NSL::imag(res), NSL::RealTypeOf<TestType>(0.),std::numeric_limits<TestType>::digits10-1) );
}
