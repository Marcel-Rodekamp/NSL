#include "../../test.hpp"
#include "LinAlg/mat_vec.tpp"
#include "complex.hpp"

/*! \file test_CG.hpp
 *
 * This file tests the the application of CG  
 * 
 * As the CG works on symetric & positive definite matrices these tests only
 * include solves for \f$M^\dagger M\f$ or \f$MM^\dagger\f$ in case of the fermion matrix
 * or $M^\dagger + M$ for a random matrix.
 *
 * These tests do not require to pass every run, as round of errors provide
 * a big issue in the CG algorithm.
 * */

template<typename Type>
void test_CG_randomMatrix(const typename NSL::RT_extractor<Type>::type eps, NSL::size_t V);

// Notice it is a good idea to check that the `FermionMatrix`
// is a viable Fermion Matrix, i.e. if it dervives from 
// `NSL::FermionMatrix::FermionMatrix`. 
// However, this is done savely in the construction of the CG regardless.
template<typename Type, class FermionMatrix>
void test_CG_fermionMatrix(FermionMatrix & M,const typename NSL::RT_extractor<Type>::type eps, NSL::size_t Nt, NSL::size_t Nx);

// =======================================================================
// Test Cases
// =======================================================================

FLOAT_NSL_TEST_CASE("CG - Random Matrix", "[CG,Random Matrix]"){
    NSL::size_t V = GENERATE(1,2,4,8,10,16,20,30,32);
    typename NSL::RT_extractor<TestType>::type eps = GENERATE(1e-1,1e-2,1e-3,1e-4);
    test_CG_randomMatrix<TestType>(eps, V);
    
    // for double types we can demand higer precisions and volumes
    if constexpr( std::is_same_v<double, typename NSL::RT_extractor<TestType>::type>) {
        NSL::size_t V_d = GENERATE(40,50,60,64,70);
        typename NSL::RT_extractor<TestType>::type eps_d = GENERATE(1e-6,1e-8,1e-10);
        test_CG_randomMatrix<TestType>(eps_d, V_d);
    } 
    
}

/*
COMPLEX_NSL_TEST_CASE("CG - Hubbard Exp - Complete", "[CG,Hubbard Exp,Complete]"){
    NSL::size_t Nt = GENERATE(1,2,8,32);
    NSL::size_t Nx = GENERATE(1,2,8,32);
    TestType beta = GENERATE(1,4,10);
    typename NSL::RT_extractor<TestType>::type eps = GENERATE(1e-4,1e-6);

    NSL::Lattice::Complete<TestType> lat(Nx);
    NSL::Tensor<TestType> phi(Nt,Nx);phi.rand();
    NSL::FermionMatrix::HubbardExp M(lat,phi,beta);
    test_CG_fermionMatrix<TestType>(M,eps,Nt,Nx);
}

COMPLEX_NSL_TEST_CASE("CG - Hubbard Exp - Ring", "[CG,Hubbard Exp,Ring]"){
    NSL::size_t Nt = GENERATE(1,2,8,32);
    NSL::size_t Nx = GENERATE(1,2,8,32);
    TestType beta = GENERATE(1,4,10);
    typename NSL::RT_extractor<TestType>::type eps = GENERATE(1e-4,1e-6);

    NSL::Lattice::Ring<TestType> lat(Nx);
    NSL::Tensor<TestType> phi(Nt,Nx);phi.rand();
    NSL::FermionMatrix::HubbardExp M(lat,phi,beta);
    test_CG_fermionMatrix<TestType>(M,eps,Nt,Nx);
}

COMPLEX_NSL_TEST_CASE("CG - Hubbard Exp - Square 1D", "[CG,Hubbard Exp,Square,1D]"){
    NSL::size_t Nt = GENERATE(1,2,8,32);
    NSL::size_t Nx = GENERATE(1,2,8,32);
    TestType beta = GENERATE(1,4,10);
    typename NSL::RT_extractor<TestType>::type eps = GENERATE(1e-4,1e-6);

    NSL::Lattice::Square<TestType> lat({Nx});
    NSL::Tensor<TestType> phi(Nt,Nx);phi.rand();
    NSL::FermionMatrix::HubbardExp M(lat,phi,beta);
    test_CG_fermionMatrix<TestType>(M,eps,Nt,Nx);
}

COMPLEX_NSL_TEST_CASE("CG - Hubbard Exp - Square 2D", "[CG,Hubbard Exp,Square,2D]"){
    NSL::size_t Nt = GENERATE(1,2,8,32);
    NSL::size_t Nx1 = GENERATE(1,2,8,32);
    NSL::size_t Nx2 = GENERATE(1,2,8,32);
    NSL::size_t Nx = Nx1*Nx2;
    TestType beta = GENERATE(1,4,10);
    typename NSL::RT_extractor<TestType>::type eps = GENERATE(1e-4,1e-6);

    NSL::Lattice::Square<TestType> lat({Nx1,Nx2});
    NSL::Tensor<TestType> phi(Nt,Nx);phi.rand();
    NSL::FermionMatrix::HubbardExp M(lat,phi,beta);
    test_CG_fermionMatrix<TestType>(M,eps,Nt,Nx);
}

COMPLEX_NSL_TEST_CASE("CG - Hubbard Exp - Square 3D", "[CG,Hubbard Exp,Square,3D]"){
    NSL::size_t Nt = GENERATE(1,2,8,32);
    NSL::size_t Nx1 = GENERATE(1,2,8,32);
    NSL::size_t Nx2 = GENERATE(1,2,8,32);
    NSL::size_t Nx3 = GENERATE(1,2,8,32);
    NSL::size_t Nx = Nx1*Nx2*Nx3;
    TestType beta = GENERATE(1,4,10);
    typename NSL::RT_extractor<TestType>::type eps = GENERATE(1e-4,1e-6);

    NSL::Lattice::Square<TestType> lat({Nx1,Nx2,Nx3});
    NSL::Tensor<TestType> phi(Nt,Nx);phi.rand();
    NSL::FermionMatrix::HubbardExp M(lat,phi,beta);
    test_CG_fermionMatrix<TestType>(M,eps,Nt,Nx);
}

COMPLEX_NSL_TEST_CASE("CG - Hubbard Exp - Square 4D", "[CG,Hubbard Exp,Square,4D]"){
    NSL::size_t Nt = GENERATE(1,2,8,32);
    NSL::size_t Nx1 = GENERATE(1,2,8,32);
    NSL::size_t Nx2 = GENERATE(1,2,8,32);
    NSL::size_t Nx3 = GENERATE(1,2,8,32);
    NSL::size_t Nx4 = GENERATE(1,2,8,32);
    NSL::size_t Nx = Nx1*Nx2*Nx3*Nx4;
    TestType beta = GENERATE(1,4,10);
    typename NSL::RT_extractor<TestType>::type eps = GENERATE(1e-4,1e-6);

    NSL::Lattice::Square<TestType> lat({Nx1,Nx2,Nx3,Nx4});
    NSL::Tensor<TestType> phi(Nt,Nx);phi.rand();
    NSL::FermionMatrix::HubbardExp M(lat,phi,beta);
    test_CG_fermionMatrix<TestType>(M,eps,Nt,Nx);
}
*/

// ======================================================================
// Implementation details: test_CG_randomMatrix
// ======================================================================

template<typename Type>
void test_CG_randomMatrix(const typename NSL::RT_extractor<Type>::type eps, NSL::size_t V){
    INFO( std::string("V = ") + std::to_string(V) );
    INFO( std::string("eps = ") + std::to_string(eps) );
    INFO( std::string("Matching Digits = ") + std::to_string(getMatchingDigits(eps)) );

    NSL::Tensor<Type> b(V);b.rand();

    NSL::Tensor<Type> mat(V,V);mat.rand();
    if (NSL::is_complex<Type>()){
        // hermitian => transpose and complex conjugate
        mat = 0.5*(mat.H() + mat);
    } else {
        // symmetric => transpose
        mat = 0.5*(mat.T() + mat);
    }

    NSL::LinAlg::CG<Type> cg( 
        [&mat](const NSL::Tensor<Type> & x) -> NSL::Tensor<Type> {
            return NSL::LinAlg::mat_vec(mat,x);
        } , 
        eps
    );

    NSL::Tensor<Type> res = cg(b);

    REQUIRE(almost_equal(NSL::LinAlg::mat_vec(mat,res),b,getMatchingDigits(eps)).all());

}

// ======================================================================
// Implementation details: test_CG_fermionMatrix
// ======================================================================
template<typename Type, class FermionMatrix>
void test_CG_fermionMatrix(FermionMatrix & M,const typename NSL::RT_extractor<Type>::type eps, NSL::size_t Nt, NSL::size_t Nx){
    NSL::Tensor<Type> b(Nt,Nx);b.rand();

    // for now we just test the default case of cg:
    // maxIter = 10000
    NSL::LinAlg::CG cg(M,NSL::FermionMatrix::MdaggerM, eps);

    NSL::Tensor<Type> res = cg(b);

    INFO( std::string("Nt = ") + std::to_string(Nt) );
    INFO( std::string("Nx = ") + std::to_string(Nx) );

    REQUIRE(almost_equal(M.MdaggerM(res),b).all());
}






