#include "../test.hpp"
#include "NSL.hpp"

template<NSL::Concept::isComplex Type>
void test_Tensor_mat_exp(NSL::size_t Nx);
template<NSL::Concept::isComplex Type>
void test_Tensor_mat_exp_pauli();

COMPLEX_NSL_TEST_CASE("Tensor 2D Matrix Exponentiation", "[Tensor,2D,Matrix Exponentiation]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32,64,128);
    test_Tensor_mat_exp<TestType>(size0);
}

COMPLEX_NSL_TEST_CASE("Tensor 2D Pauli-Matrix Exponentiation", "[Tensor,2D,Pauli Matrix Exponentiation]"){
    test_Tensor_mat_exp_pauli<TestType>();
}

//=======================================================================
// Implementation Details: test_Tensor_mat_exp
//=======================================================================
template<NSL::Concept::isComplex Type>
void test_Tensor_mat_exp(NSL::size_t Nx){

    // This tests to see if exp(A)*exp(-A) = I for a random Hermitian A normalized to a norm of order 1.
    // Without the 1/Nx the largest eigenvalue of (R + R^H)/2 (entries of R uniform in [0,1)) is ~Nx/2: then
    // exp(A)*exp(-A) cancels terms of size exp(Nx/2) down to 1, which no floating-point product can do for Nx >~ 30
    // (cond(exp(A)) = exp(2||A||)); that is not a property of mat_exp.
    INFO(fmt::format("Nx={}",Nx));
    NSL::Tensor<Type> A(Nx,Nx); A.rand();
    A += A.H();
    A /= static_cast<NSL::RealTypeOf<Type>>(2*Nx);
 
    NSL::Tensor<Type> B(-1*A,true);

    A.mat_exp();  // exp(A)
    B.mat_exp();  // exp(-A)

    NSL::Tensor<Type> C = NSL::LinAlg::mat_mul(A,B);  // exp(A)*exp(-A) = I

    REQUIRE( 
        almost_equal(
            C-NSL::Matrix::Identity<Type>(Nx),
            Type(0),
            std::numeric_limits<Type>::digits10-2
        ).all() 
    );
}

//=======================================================================
// Implementation Details: test_Tensor_mat_exp_pauli
//=======================================================================
template<NSL::Concept::isComplex Type>
void test_Tensor_mat_exp_pauli(){
    // This tests the relation exp(i theta (\hat{n}\cdot \vec{\sigma})) = I cos(theta)+ i(\hat{n}\cdot \vec{\sigma}) sin(theta)

    // first define the sigma matrices. . .
    NSL::Tensor<Type> sigma1(2,2);
    NSL::Tensor<Type> sigma2(2,2);
    NSL::Tensor<Type> sigma3(2,2);
    sigma1(0,1)=NSL::complex<NSL::RealTypeOf<Type>>{1,0};    sigma1(1,0)=NSL::complex<NSL::RealTypeOf<Type>>{1,0};
    sigma2(0,1)=NSL::complex<NSL::RealTypeOf<Type>>{0,-1};   sigma2(1,0)=NSL::complex<NSL::RealTypeOf<Type>>{0,1};
    sigma3(0,0)=NSL::complex<NSL::RealTypeOf<Type>>{1,0};    sigma3(1,1)=NSL::complex<NSL::RealTypeOf<Type>>{-1,0};

    // now I construct a random unit vector. . .
    NSL::Tensor<NSL::RealTypeOf<Type>> n(3);
    n.rand();
    n /= sqrt(n(0)*n(0)+n(1)*n(1)+n(2)*n(2));  // normalized vector
                                               
    INFO(n);

    // here is random angle
    NSL::RealTypeOf<Type> theta = NSL::Tensor<NSL::RealTypeOf<Type>>(1).rand()(0) * 6.28;
    
    NSL::Tensor<Type> expSigma(2,2);
    NSL::complex<NSL::RealTypeOf<Type>> itheta(0,theta);
    NSL::complex<NSL::RealTypeOf<Type>> isintheta(0,sin(theta));

    expSigma = itheta * (n(0)*sigma1 + n(1)*sigma2 + n(2)*sigma3);
    expSigma = expSigma.mat_exp(); // exp(i*theta*(n.sigma))
    
    NSL::Tensor<Type> answer(2,2);
    auto I2 = NSL::Matrix::Identity<Type>(2);
    answer = cos(theta)*I2 + isintheta*(n(0)*sigma1 + n(1)*sigma2 + n(2)*sigma3); // cos(theta)*I + i*sin(theta)*(n.sigma)
    INFO(expSigma);
    INFO(answer);
    REQUIRE( almost_equal(expSigma-answer,Type(0),std::numeric_limits<Type>::digits10-8).all() );
    
}

