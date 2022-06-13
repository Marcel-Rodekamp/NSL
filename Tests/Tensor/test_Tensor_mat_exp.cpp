#include "../test.hpp"
#include "NSL.hpp"

template<NSL::Concept::isComplex Type>
void test_Tensor_mat_exp(NSL::size_t Nx);

COMPLEX_NSL_TEST_CASE("Tensor 2D Matrix Exponentiation", "[Tensor,2D,Matrix Exponentiation]"){
  NSL::size_t size0 = GENERATE(1,2,4,8,16,32,64);
  test_Tensor_mat_exp<TestType>(size0);
}

//=======================================================================
// Implementation Details: test_Tensor_mat_exp
//=======================================================================
template<NSL::Concept::isComplex Type>
void test_Tensor_mat_exp(NSL::size_t Nx){


  // This tests to see if exp(A)*exp(-A) = I
  INFO(Nx);
  NSL::Tensor<Type> A(Nx,Nx);
  A.rand();
  Type * addr_bak = A.data();
  NSL::Tensor<Type> B(-1*A,true);
  NSL::Tensor<Type> C(Nx,Nx);

  A.mat_exp();  // exp(A)
  B.mat_exp();  // exp(-A)
  C = NSL::LinAlg::mat_mul(A,B);  // exp(A)*exp(-A) = I

  auto I = NSL::Matrix::Identity<Type>(Nx);
  //INFO(C);
  REQUIRE( almost_equal(C-I,Type(0),std::numeric_limits<Type>::digits10-12).all() );
  // NOTE!!!! This tolerance is quite low!  There is something suspicious about torch's mat_exp routine!!!!!


  if (Nx == 2) {
    // This tests the relation exp(i theta (\hat{n}\cdot \vec{\sigma})) = I cos(theta)+ i(\hat{n}\cdot \vec{\sigma}) sin(theta)

    // first define the sigma matrices. . .
    NSL::Tensor<Type> sigma1(2,2);
    NSL::Tensor<Type> sigma2(2,2);
    NSL::Tensor<Type> sigma3(2,2);
    sigma1(0,1)=NSL::complex<typename NSL::RT_extractor<Type>::type>{1,0};    sigma1(1,0)=NSL::complex<typename NSL::RT_extractor<Type>::type>{1,0};
    sigma2(0,1)=NSL::complex<typename NSL::RT_extractor<Type>::type>{0,-1};   sigma2(1,0)=NSL::complex<typename NSL::RT_extractor<Type>::type>{0,1};
    sigma3(0,0)=NSL::complex<typename NSL::RT_extractor<Type>::type>{1,0};    sigma3(1,1)=NSL::complex<typename NSL::RT_extractor<Type>::type>{-1,0};

    // now I construct a random unit vector. . .
    NSL::Tensor<Type> n(3);
    n.rand();
    n.imag() = static_cast<typename NSL::RT_extractor<Type>::type>(0);
    n /= sqrt(n(0)*n(0)+n(1)*n(1)+n(2)*n(2));  // normalized vector

    // here is random angle
    double theta = std::rand();
    
    NSL::Tensor<Type> expSigma(2,2);
    NSL::complex<typename NSL::RT_extractor<Type>::type> tmp(0,theta);
    NSL::complex<typename NSL::RT_extractor<Type>::type> sintheta(0,sin(theta));
    expSigma = tmp * (n(0)*sigma1 + n(1)*sigma2 + n(2)*sigma3);
    expSigma = expSigma.mat_exp(); // exp(i*theta*(n.sigma))
  
    NSL::Tensor<Type> answer(2,2);
    auto I2 = NSL::Matrix::Identity<Type>(2);
    answer = cos(theta)*I2 + sintheta*(n(0)*sigma1 + n(1)*sigma2 + n(2)*sigma3); // cos(theta)*I + i*sin(theta)*(n.sigma)
    INFO(expSigma);
    INFO(answer);
    REQUIRE( almost_equal(expSigma-answer,Type(0),std::numeric_limits<Type>::digits10-8).all() );
  }
    
}


