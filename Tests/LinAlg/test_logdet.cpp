#include "complex.hpp"
#include "../test.hpp"
#include <typeinfo>
#include "Tensor/tensor.hpp"
#include "LinAlg/mat_conj.hpp"
#include "LinAlg/det.hpp"


using size_type = int64_t;

template<typename T>
void test_logdet(const size_type size) {


    //setting precision
    auto limit =  100*std::numeric_limits<T>::epsilon();

    NSL::TimeTensor<NSL::complex<T>> a(size,size);    
    a.rand();

    //Testing if logdet(a_dagger) = (logdet(a))*
    REQUIRE(fabs((NSL::LinAlg::det(a)).real() - (NSL::LinAlg::det(NSL::LinAlg::adjoint(a))).real()) <= limit);
    REQUIRE(fabs((NSL::LinAlg::det(a)).imag() + (NSL::LinAlg::det(NSL::LinAlg::adjoint(a))).imag()) <= limit);


}

//Test Cases

REAL_NSL_TEST_CASE( "LinAlg: logdet", "[LinAlg, logdet]" ) {
    
    const size_type size = GENERATE(2, 4, 8, 10, 12, 14, 16);
    
    test_logdet<TestType>(size);
    
}   