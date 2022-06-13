/*!
 * this tests the different versions of `NSL::LinAlg::shift`
 * \todo Not fully developed!
 * */

#include "../test.hpp"

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral SizeTypes>
void test_linalg_mat_vec(SizeTypes size);

FLOAT_NSL_TEST_CASE("LinAlg MatVec", "[LinAlg,mat_vec]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    test_linalg_mat_vec<TestType>(size0);
}


// ======================================================================
// Implementation details: shiftTensor
// ======================================================================

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral SizeTypes>
void test_linalg_mat_vec(SizeTypes size){
    
    // Forgive 2 decimal places of floating-point precision loss.
    auto limit = std::pow(10, 2-std::numeric_limits<Type>::digits10);
    
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
    for(int i=0; i<size; i++){
	Type tmp = 0;
	for(int j=0; j<size; j++){
	    tmp += A(i,j)*x(j);
	}
	auto res = NSL::LinAlg::abs(b(i)-tmp);
	REQUIRE( res <= limit );
    }

    
    // Check additivity: (A+B).(x+y) = A.x + A.y + B.x + B.y
    // =====================================================
    // perform the multiplication
    NSL::Tensor<Type> B(size,size);
    NSL::Tensor<Type> y(size);

    B.rand();
    y.rand();
    
    b = NSL::LinAlg::mat_vec(A+B,x+y);
    NSL::Tensor<Type> c = NSL::LinAlg::mat_vec(A,x) + NSL::LinAlg::mat_vec(A,y) + NSL::LinAlg::mat_vec(B,x) + NSL::LinAlg::mat_vec(B,y);

    REQUIRE( almost_equal(b,c).all() );
    
}



