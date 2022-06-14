#include "../test.hpp"

template<NSL::Concept::isNumber Type>
void test_linalg_mat_vec(NSL::size_t size);

template<NSL::Concept::isNumber Type>
void test_mat_vec_linearity(NSL::size_t size);

FLOAT_NSL_TEST_CASE("LinAlg mat_vec", "[LinAlg,mat_vec]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32,1024);
    test_linalg_mat_vec<TestType>(size0);
}

FLOAT_NSL_TEST_CASE("LinAlg mat_vec linearity", "[LinAlg,mat_vec,linearity]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32,64,128,256,512,1024);
    test_mat_vec_linearity<TestType>(size0);
}


template<NSL::Concept::isNumber Type>
void test_linalg_mat_vec(NSL::size_t size){
    
    //create necessary Tensors
    NSL::Tensor<Type> A(size,size);
    NSL::Tensor<Type> x(size);

    A.rand();
    x.rand();

    // make sure rand is not just feeding zeros
    REQUIRE ( NSL::LinAlg::abs(A(0,0))      > 0.0  );
    REQUIRE ( NSL::LinAlg::abs(x(0))        > 0.0  );
    REQUIRE ( NSL::LinAlg::abs(A(0,0)-x(0)) > 0.0  );

    // Basic check
    // ===========
    // perform the multiplication
    NSL::Tensor<Type> b = NSL::LinAlg::mat_vec(A,x);

    // And now on foot
    NSL::Tensor<Type> tmp(size);
    for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
            tmp(i) += A(i,j)*x(j);
        }
    }

    REQUIRE( almost_equal(tmp, b).all() );
}
    
template<NSL::Concept::isNumber Type>
void test_mat_vec_linearity(NSL::size_t size){

    // Check additivity: (A+B).(x+y) = A.x + A.y + B.x + B.y
    // =====================================================
    // perform the multiplication
    NSL::Tensor<Type> A(size,size); A.rand();
    NSL::Tensor<Type> B(size,size); B.rand();
    NSL::Tensor<Type> x(size);      x.rand();
    NSL::Tensor<Type> y(size);      y.rand();

    NSL::Tensor<Type> b = NSL::LinAlg::mat_vec(A+B,x+y);
    NSL::Tensor<Type> c = NSL::LinAlg::mat_vec(A,x) + NSL::LinAlg::mat_vec(A,y) + NSL::LinAlg::mat_vec(B,x) + NSL::LinAlg::mat_vec(B,y);

    INFO(size);

    int precision = std::numeric_limits<Type>::digits10;
    if(size > 128){
        precision -= 1;
    }

    REQUIRE( almost_equal(b,c, precision).all() );
    
}



