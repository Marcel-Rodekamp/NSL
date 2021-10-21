#include "../test.hpp"
#include "Tensor/tensor.hpp"

//! \file Tests/Tensor/test_tensor.cpp

// Torch requirement
using size_type = int64_t;

template<typename T, typename ... SizeTypes>
void test_constructor(const SizeTypes &... sizes);

template<typename T, typename ... SizeTypes>
void test_access(const SizeTypes &... sizes);

template<typename T, typename ... SizeTypes>
void test_slice(const SizeTypes &... sizes);

template<typename T, typename ... SizeTypes>
void test_expand(const size_type & newSize, const SizeTypes &... sizes);

template<typename T, typename ... SizeTypes>
void test_shift(const size_type & shift, const SizeTypes &... sizes);

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
// Expand
// =============================================================================

NSL_TEST_CASE("Tensor: Expand 1D -> 2D", "[Tensor,Expand]") {
    size_type newSize = GENERATE(1,10,100);
    size_type size0 = GENERATE(1,10,100);
    test_expand<TestType>(newSize,size0);
}

NSL_TEST_CASE("Tensor: Expand 2D -> 3D", "[Tensor,Expand]") {
    size_type newSize = GENERATE(1,5,10);
    size_type size0 = GENERATE(1,10,20);
    size_type size1 = GENERATE(1,10,20);
    test_expand<TestType>(newSize,size0,size1);
}

NSL_TEST_CASE("Tensor: Expand 3D -> 4D", "[Tensor,Expand]") {
    size_type newSize = GENERATE(1,5,10);
    size_type size0 = GENERATE(1,5,10);
    size_type size1 = GENERATE(1,5,10);
    size_type size2 = GENERATE(1,5,10);
    test_expand<TestType>(newSize,size0,size1,size2);
}


// =============================================================================
// Shift
// =============================================================================

NSL_TEST_CASE("Tensor: Shift 1D", "[Tensor,Shift]") {
    size_type shift = GENERATE(1,2,3,4);
    size_type size0 = GENERATE(5,100,1000);
    test_shift<TestType>(shift,size0);

    // corner cases
    test_shift<TestType>(shift,shift);
    test_shift<TestType>(shift,shift-1);
}

NSL_TEST_CASE("Tensor: Shift 2D", "[Tensor,Shift]") {
    size_type shift = GENERATE(1,2,3,4);
    size_type size0 = GENERATE(5,10,100);
    size_type size1 = GENERATE(5,10,100);

    test_shift<TestType>(shift,size0,size1);

    // corner cases
    test_shift<TestType>(shift,shift,size1);
    test_shift<TestType>(shift,shift-1,size1);

    test_shift<TestType>(shift,size0,shift);
    test_shift<TestType>(shift,size0,shift-1);
}

NSL_TEST_CASE("Tensor: Shift 3D", "[Tensor,Shift]") {
    size_type shift = GENERATE(1,2,3,4);
    size_type size0 = GENERATE(5,10,100);
    size_type size1 = GENERATE(5,10,100);
    size_type size2 = GENERATE(5,10,100);

    test_shift<TestType>(shift,size0,size1,size2);

    // corner cases
    test_shift<TestType>(shift,shift,size1,size2);
    test_shift<TestType>(shift,shift-1,size1,size2);

    test_shift<TestType>(shift,size0,shift,size2);
    test_shift<TestType>(shift,size0,shift-1,size2);

    test_shift<TestType>(shift,size0,size1,shift);
    test_shift<TestType>(shift,size0,size1,shift-1);
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
 *  * `Tensor::shape()`
 *  * `Tensor::shape(const size_t &)`
 * */
template<typename T, typename ... SizeTypes>
void test_constructor(const SizeTypes &... sizes){
    INFO("Type = " + std::string(typeid(T).name()));
    INFO("Dimension = " + std::to_string(sizeof...(SizeTypes)));
    INFO(("Sizes = " + ... + (" " + std::to_string(sizes))));

    std::array<size_type,sizeof...(sizes)> size_array{{sizes...}};

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

    // shape like constructor
    NSL::Tensor<T> D(A.shape());
    REQUIRE(D.dim() == sizeof...(sizes));
    for(int d = 0; d < sizeof...(sizes); ++d){
        REQUIRE(D.shape(d) == size_array[d]);
        REQUIRE((D == static_cast<T>(0)).all());
    }
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

//! Check the expand operation
/*!
 * Tests:
 *
 * * `Tensor<Type,RealType>::expand(const size_t & newSize)`
 *
 * \n
 * Requires:
 *
 *  * `Tensor<Type,RealType>::dim()`
 *  * `Tensor<Type,RealType>::shape()`
 *  * `Tensor<Type,RealType>::shape(const size_t & dim)`
 *  * `Tensor<Type,RealType>::operator=(Tensor<Type,RealType>)`
 *  * `Tensor<Type,RealType>::numel()`
 * */
template <typename T, typename ... SizeTypes>
void test_expand(const size_type & newSize, const SizeTypes &... sizes){
    INFO("Type = " + std::string(typeid(T).name()));
    INFO("Dimension = " + std::to_string(sizeof...(SizeTypes)));
    INFO(("Sizes = " + ... + (" " + std::to_string(sizes))));
    std::array<size_type,sizeof...(sizes)> sizes_array{{sizes...}};

    // create tensor
    NSL::Tensor<T> A(sizes...);

    // fill the tensor with values and backup it
    A = static_cast<T>(1);
    NSL::Tensor<T> Abak(A);

    // expand the tensor
    A.expand(newSize);

    // check dimensionality
    REQUIRE(A.dim() == sizeof...(sizes) + 1);
    // check sizes
    for(size_type d = 0; d < sizeof...(sizes); ++d){
        REQUIRE(A.shape(d) == sizes_array[d]);
    }
    REQUIRE(A.shape(sizeof...(sizes)) == newSize);

    // each slice of the new dimension contains the same data of the old array
    for(size_type i = 0; i < newSize; ++i){
        if constexpr (sizeof...(sizes) == 1){
            for(size_type x0 = 0; x0 < sizes_array[0]; ++x0){
                REQUIRE(A(x0,i) == Abak(x0));
            }
        } else if constexpr (sizeof...(sizes) == 2) {
            for(size_type x0 = 0; x0 < sizes_array[0]; ++x0){
                for(size_type x1 = 0; x1 < sizes_array[1];++x1) {
                    REQUIRE(A(x0, x1, i) == Abak(x0,x1));
                }
            }
        } else if constexpr(sizeof...(sizes) == 3){
            for(size_type x0 = 0; x0 < sizes_array[0]; ++x0){
                for(size_type x1 = 0; x1 < sizes_array[1];++x1) {
                    for(size_type x2 = 0; x2 < sizes_array[2];++x2) {
                        REQUIRE(A(x0, x1, x2, i) == Abak(x0,x1,x2));
                    }
                }
            }
        } else {
            INFO("Error: test_expand is only implemented for up to 3 dimensions");
            REQUIRE(false);
        }
    }

}


//! Check the shift operation
/*!
 * Tests:
 *
 * * `Tensor<Type,RealType>::shift(const size_t &)`
 * * `Tensor<Type,RealType>::shift(const size_t & const )`
 * * `Tensor<Type,RealType>::shift(const size_t &)`
 * * `Tensor<Type,RealType>::shift(const size_t &)`
 *
 * \n
 * Requires:
 *
 *  * `Tensor<Type,RealType>::slice(size_t,size_t,size_t)`
 *  * `Tensor<Type,RealType>::operator==(Tensor<Type,RealType>)`
 *  * `Tensor<Type,RealType>::operator=(Tensor<Type,RealType>)`
 *  * `Tensor<Type,RealType>::all()`
 * */
template <typename T, typename ... SizeTypes>
void test_shift(const size_type & shift, const SizeTypes &... sizes){
    INFO("Type = " + std::string(typeid(T).name()));
    INFO("Dimension = " + std::to_string(sizeof...(SizeTypes)));
    INFO("Shift = " + std::to_string(shift));
    INFO(("Sizes = " + ... + (" " + std::to_string(sizes))));
    std::array<size_type, sizeof...(sizes)> size_array{{sizes...}};

    // create a Tensor
    NSL::Tensor<T> A(sizes...);

    if constexpr(std::is_same<T,bool>()) {
        for(size_type i = 0; i < A.numel(); ++i){
            A.data()[i] = static_cast<bool>(i % 2);
        }
    } else if constexpr(std::is_same<T,int>()){
        for(size_type i = 0; i < A.numel(); ++i){
            A.data()[i] = i;
        }
    } else {
        A.rand();
    }

    NSL::Tensor<T> Abak(A);

    // shift 0th axis by shift elements
    A.shift(shift);

    for(size_type i = 0; i < size_array[0]; ++i){
        REQUIRE((Abak.slice(0,(i-shift+size_array[0])%size_array[0],(i-shift+1+size_array[0])%size_array[0]) == A.slice(0,i,i+1)).all());
    }

    A = Abak;

    // shift dth axis by shift element
    for(size_type d = 0; d < sizeof...(sizes); ++d){
        A.shift(shift,d);

        for(size_type i = 0; i < size_array[d]; ++i){
            REQUIRE((Abak.slice(d,(i-shift+size_array[d])%size_array[d],(i-shift+1+size_array[d])%size_array[d]) == A.slice(d,i,i+1)).all());
        }

        A = Abak;
    }

    // the following test apply different boundary conditions as this does not make
    // sense in the bool case we exit here
    if constexpr(std::is_same<T,bool>()){
        return;
    }

    // generally if a boundary condition is applied:
    // Psi(Nt) = boundary * Psi(0)
    // Psi(Nt+1) = boundary * Psi(1)
    // ...
    // Psi(Nt+shift-1) = boundary * Psi(shift-1)
    T tmp_b = static_cast<T>(1);

    // test anti-periodic boundary condition: boundary = -1
    // shift 0th axis by shift elements
    A.shift(shift,static_cast<T>(-1));

    for(size_type i = 0; i < size_array[0]; ++i){
        if(i < shift){ // shift = 1: Psi(Nt) == - Psi(0)
            tmp_b = static_cast<T>(-1);
        }
        REQUIRE((Abak.slice(0,(i-shift+size_array[0])%size_array[0],(i-shift+1+size_array[0])%size_array[0])*tmp_b == A.slice(0,i,i+1)).all());
        if(i < shift){
            tmp_b = static_cast<T>(1);
        }
    }

    A = Abak;

}
