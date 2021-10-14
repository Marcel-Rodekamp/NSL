#include "../test.hpp"
#include "Tensor/tensor.hpp"

//! \file Tests/Tensor/test_tensor.cpp

// Torch requirement
using size_type = long int;

template<typename T, typename ... SizeTypes>
void test_constructor(const SizeTypes &... sizes);

template<typename T, typename ... SizeTypes>
void test_access(const SizeTypes &... sizes);

template<typename T, typename ... SizeTypes>
void test_slice(const SizeTypes &... sizes);

template class NSL::Tensor<NSL::complex<float>>;

// =============================================================================
// Constructors
// =============================================================================

NSL_TEST_CASE("Tensor: Constructor 1D", "[Tensor,Constructor]") {
    size_t size0 = GENERATE(1,100,1000);
    test_constructor<TestType>(size0);
}


NSL_TEST_CASE("Tensor: Constructor 2D", "[Tensor,Constructor]") {
    size_t size0 = GENERATE(1, 10, 100);
    size_t size1 = GENERATE(1, 10, 100);
    test_constructor<TestType>(size0, size1);
}


NSL_TEST_CASE("Tensor: Constructor 3D", "[Tensor,Constructor]"){
    size_t size0 = GENERATE(1, 10, 100);
    size_t size1 = GENERATE(1, 10, 100);
    size_t size2 = GENERATE(1, 10, 100);
    test_constructor<TestType>(size0, size1, size2);
}


// =============================================================================
// Random Access
// =============================================================================

NSL_TEST_CASE("Tensor: Element Access 1D", "[Tensor,Access]") {
    size_t size0 = GENERATE(1,100,1000);
    test_access<TestType>(size0);
}


NSL_TEST_CASE("Tensor: Element Access 2D", "[Tensor,Access]") {
    size_t size0 = GENERATE(1, 10, 100);
    size_t size1 = GENERATE(1, 10, 100);
    test_access<TestType>(size0, size1);
}


NSL_TEST_CASE("Tensor: Element Access 3D", "[Tensor,Access]"){
    size_t size0 = GENERATE(1, 10, 100);
    size_t size1 = GENERATE(1, 10, 100);
    size_t size2 = GENERATE(1, 10, 100);
    test_access<TestType>(size0, size1, size2);
}


// =============================================================================
// Random Access
// =============================================================================

NSL_TEST_CASE("Tensor: Slice 1D", "[Tensor,Access]") {
    size_t size0 = GENERATE(1,100,1000);
    test_slice<TestType>(size0);
}


NSL_TEST_CASE("Tensor: Slice 2D", "[Tensor,Access]") {
    size_t size0 = GENERATE(1, 10, 100);
    size_t size1 = GENERATE(1, 10, 100);
    test_slice<TestType>(size0, size1);
}


NSL_TEST_CASE("Tensor: Slice 3D", "[Tensor,Access]"){
    size_t size0 = GENERATE(1, 10, 100);
    size_t size1 = GENERATE(1, 10, 100);
    size_t size2 = GENERATE(1, 10, 100);
    test_slice<TestType>(size0, size1, size2);
}


// =============================================================================
// Implementations
// =============================================================================

//! Check the constructors
/*!
 * Tests:
 *
 *  * `Tensor<Type,RealType>::Tensor(size_t size0, SizeType... sizes)`
 *  * `Tensor<Type,RealType>::Tensor(Tensor<Type,RealType> & other)`
 *  * `Tensor<Type,RealType>::Tensor(Tensor<Type,RealType> && other)`
 *
 * \n
 * Requires:
 *
 *  * `Tensor::operator==(const Type & value)`
 *  * `Tensor::operator=(const Type & value)`
 *  * `Tensor::all()`
 * */
template<typename T, typename ... SizeTypes>
void test_constructor(const SizeTypes &... sizes){
    INFO("Type = " + std::string(typeid(T).name()));
    INFO("Dimension = " + std::to_string(sizeof...(SizeTypes)));
    INFO(("Sizes = " + ... + (" " + std::to_string(sizes))));

    // D-dimensional constructor
    NSL::Tensor<T> A(sizes...);
    REQUIRE((A==static_cast<T>(0)).all());
    A = static_cast<T>(1);

    // copy constructor
    NSL::Tensor<T> B(A);
    REQUIRE((B==static_cast<T>(1)).all());

    // move constructor
    NSL::Tensor<T> C(std::move(B));
    // check that all elements are correctly available
    REQUIRE((C==static_cast<T>(1)).all());
    // check that the original "has been moved"
    REQUIRE_THROWS((B==static_cast<T>(1)).all());
}


//! Check the access methods
/*!
 * Tests:
 *
 *  * `Tensor<Type,RealType>::operator(const Args &... sizes)()`
 *  * `Tensor<Type,RealType>::operator(const Args &... sizes)() const`
 *  * `Tensor<Type,RealType>::data()`
 *  * `Tensor<Type,RealType>::data() const`
 *
 * \n
 * Requires:
 *
 *  * `Tensor<Type,RealType>::Tensor(size_t size0, SizeType... sizes)`
 *  * For d-dimensional with \f$d>1\f$, assume naive memory layout:
 *      \f[
 *      \text{LinearIndex} \left[(i_0,i_1,\dots),(\texttt{sizes}\dots)\right] = \sum_{n_d=0}^{d-1} \left[ i_{n_d}\prod_{m=n_d+1}^{d-1} \texttt{sizes}\left[m\right]\right]
 *      \f]
 * */
template<typename T, typename ... SizeTypes>
void test_access(const SizeTypes &... sizes){
    INFO("Type = " + std::string(typeid(T).name()));
    INFO("Dimension = " + std::to_string(sizeof...(SizeTypes)));
    INFO(("Sizes = " + ... + (" " + std::to_string(sizes))));

    std::array<size_t, sizeof...(SizeTypes)> extents{{sizes...}};

    // initialize a Tensor (all values are 0)
    NSL::Tensor<T> A(sizes...);
    const NSL::Tensor<T> Aconst(sizes...);

    if constexpr (sizeof...(SizeTypes) == 1){
        for(size_t n0 = 0; n0 < extents[0]; ++n0){
            INFO("n_0 = " + std::to_string(n0));

            // Check that get element works
            REQUIRE(A(n0) == static_cast<T>(0));
            REQUIRE(A.data()[n0] == static_cast<T>(0));
            REQUIRE(Aconst(n0) == static_cast<T>(0));
            REQUIRE(Aconst.data()[n0] == static_cast<T>(0));

            // check that set element works (assuming get element works)
            A(n0) = static_cast<T>(1);
            REQUIRE(A(n0) == static_cast<T>(1));
            A.data()[n0] = static_cast<T>(2);
            REQUIRE(A(n0) == static_cast<T>(2));
        }
    } else if constexpr(sizeof...(SizeTypes) == 2){
        for(size_t n0=0,n1=0; n0 < extents[0] && n1 < extents[1]; ++n0, ++n1){
            INFO("n_0 = " + std::to_string(n0));
            INFO("n_1 = " + std::to_string(n1));

            // this assumes naive layout (e.g. torch::strided)
            size_t n = n0 * extents[1] + n1;

            // Check that get element works
            REQUIRE(A(n0,n1) == static_cast<T>(0));
            REQUIRE(A.data()[n] == static_cast<T>(0));
            REQUIRE(Aconst(n0,n1) == static_cast<T>(0));
            REQUIRE(Aconst.data()[n] == static_cast<T>(0));

            // check that set element works (assuming get element works)
            A(n0,n1) = static_cast<T>(1);
            REQUIRE(A(n0,n1) == static_cast<T>(1));
            A.data()[n] = static_cast<T>(2);
            REQUIRE(A(n0,n1) == static_cast<T>(2));
        }
    } else if constexpr(sizeof...(SizeTypes) == 3){
        for(size_t n0=0,n1=0,n2=0; n0 < extents[0] && n1 < extents[1] && n2 < extents[2]; ++n0, ++n1, ++n2){
            INFO("n_0 = " + std::to_string(n0));
            INFO("n_1 = " + std::to_string(n1));
            INFO("n_2 = " + std::to_string(n2));

            // this assumes naive layout (e.g. torch::strided)
            size_t n = n0 * extents[1] * extents[2] + n1 * extents[2] + n2;

            // Check that get element works
            REQUIRE(A(n0,n1,n2) == static_cast<T>(0));
            REQUIRE(A.data()[n] == static_cast<T>(0));
            REQUIRE(Aconst(n0,n1,n2) == static_cast<T>(0));
            REQUIRE(Aconst.data()[n] == static_cast<T>(0));

            // check that set element works (assuming get element works)
            A(n0,n1,n2) = static_cast<T>(1);
            REQUIRE(A(n0,n1,n2) == static_cast<T>(1));
            A.data()[n] = static_cast<T>(2);
            REQUIRE(A(n0,n1,n2) == static_cast<T>(2));
        }
    } else {
        INFO("Test is only developed for dimension = 1, 2, 3");
        REQUIRE(false);
    }
}

//! Check the slicing operation
/*!
 * Tests:
 *
 * * `Tensor<Type,RealType>::slice(const size_t & dim, const size_t start, const size_t & end, const size_t step = 1)`
 * * `Tensor<Type,RealType>::slice(const size_t & dim, const size_t start, const size_t & end, const size_t step = 1) const`
 *
 *
 * \n
 * Requires:
 *
 *  * `Tensor<Type,RealType>::data()`
 *  * `Tensor<Type,RealType>::shape()`
 *  * `Tensor<Type,RealType>::shape(const size_t & dim)`
 *  * `Tensor<Type,RealType>::dim()`
 *  * `Tensor<Type,RealType>::operator=(Tensor<Type,RealType>)`
 *
 * */
template<typename T, typename ... SizeTypes>
void test_slice(const SizeTypes &... sizes){
    INFO("Type = " + std::string(typeid(T).name()));
    INFO("Dimension = " + std::to_string(sizeof...(SizeTypes)));
    INFO(("Sizes = " + ... + (" " + std::to_string(sizes))));

    // D-dimensional constructor
    NSL::Tensor<T> A(sizes...);

    // fill the tensor
    const size_t N = (1 * ... * sizes);
    for(size_t n = 0; n < N; ++n){
        A.data()[n] = n;
    }

    // single dim slice
    // Expectation:
    //     * Aslice.dim() = A.dim() = sizeof...(SizeTypes)
    //     * Aslice.shape(0) = 1
    NSL::Tensor<T> Aslice = A.slice(/*dim=*/0,/*start=*/0,/*end=*/1,/*step=*/1);
    REQUIRE(Aslice.dim() == A.dim());
    REQUIRE(Aslice.shape(0) == 1);
    for(size_t d = 1; d < sizeof...(SizeTypes); ++d){
        REQUIRE(Aslice.shape(d) == A.shape(d));
    }
}
