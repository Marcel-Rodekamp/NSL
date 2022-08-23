#include "../test.hpp"

//! \file Tests/Tensor/test_randomAccess.cpp
/*!
 * To test random access, we need to ensure that all the elements of a 
 * Tensor come out correctly for one of the random access methods.
 * Therefore we demand,
 * - Given a set of indices the element must match (read access) 
 * - Given a set of indices the elements must be changed correctly (write access)
 * - Given a set of slices the reported stats must match the expected once
 * - Given a set of slices the elements must match (read access)
 * - Given a set of slices the elements must be changed correctly (write access)
 * - Requesting the (C) pointer must be consistent with the addresses 
 *   of elements
 *
 * Providing tests for:
 * - operator()(NSL::size_t ...)
 * - operator()(NSL::Slice ...)
 * - operator[](NSL::size_t)
 * - data()
 * */

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void indexAccess(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void indexWriteAccess(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void sliceAccess(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void sliceWriteAccess(SizeTypes ... sizes);

template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void mixedAccess(SizeTypes ... sizes);

NSL_TEST_CASE("Tensor 1D Random Access", "[Tensor,1D,Random Access]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    indexAccess<TestType>(size0);
    indexWriteAccess<TestType>(size0);
}

NSL_TEST_CASE("Tensor 1D Slice Access", "[Tensor,1D,Slice Access]"){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    sliceAccess<TestType>(size0);
    sliceWriteAccess<TestType>(size0);
}

NSL_TEST_CASE("Tensor 2D Random Access", "[Tensor,2D,Random Access]"){
    NSL::size_t size0 = GENERATE(1,2,4,8,16,32);
    NSL::size_t size1 = GENERATE(1,2,4,8,16,32);
    indexAccess<TestType>(size0,size1);
    indexWriteAccess<TestType>(size0,size1);
}
NSL_TEST_CASE("Tensor 2D Slice Access", "[Tensor,2D,Slice Access]"){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    NSL::size_t size1 = GENERATE(1,2,4,8);
    sliceAccess<TestType>(size0,size1);
    sliceWriteAccess<TestType>(size0,size1);
}
NSL_TEST_CASE("Tensor 2D Mixed Slice/Random Access", "[Tensor,2D,Mixed Access]"){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    NSL::size_t size1 = GENERATE(1,2,4,8);
    mixedAccess<TestType>(size0,size1);
    //sliceWriteAccess<TestType>(size0,size1);
    // MARKER1
}

NSL_TEST_CASE("Tensor 3D Random Access", "[Tensor,3D,Random Access]"){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    NSL::size_t size1 = GENERATE(1,2,4,8);
    NSL::size_t size2 = GENERATE(1,2,4,8);
    indexAccess<TestType>(size0,size1,size2);
    indexWriteAccess<TestType>(size0,size1,size2);
}
NSL_TEST_CASE("Tensor 3D Slice Access", "[Tensor,3D,Slice Access]"){
    NSL::size_t size0 = GENERATE(1,2,4,6);
    NSL::size_t size1 = GENERATE(1,2,4,6);
    NSL::size_t size2 = GENERATE(1,2,4,6);
    sliceAccess<TestType>(size0,size1,size2);
    sliceWriteAccess<TestType>(size0,size1,size2);
}

NSL_TEST_CASE("Tensor 4D Random Access", "[Tensor,4D,Random Access]"){
    NSL::size_t size0 = GENERATE(1,2,4,8);
    NSL::size_t size1 = GENERATE(1,2,4,8);
    NSL::size_t size2 = GENERATE(1,2,4,8);
    NSL::size_t size3 = GENERATE(1,2,4,8);
    indexAccess<TestType>(size0,size1,size2,size3);
    indexWriteAccess<TestType>(size0,size1,size2,size3);
}

//=======================================================================
// Implementation Details: Linearized Index
//=======================================================================
// This function encodes how the indices are mepped onto the memory.
// It is a copy (andm ust match) the one implemented for the Tensor in 
// src/NSL/Tensor/Impl/base.tpp
template<NSL::Concept::isIntegral ... SizeTypes>
NSL::size_t linearIndex(std::vector<NSL::size_t> strides, const SizeTypes &... indices){
    std::array<size_t, sizeof...(indices)> a_indices = {indices...};
    
    size_t offset = 0;
    for(size_t d = 0 ; d < sizeof...(indices); ++d){
        offset += a_indices[d] * strides[d];
    }
    return offset;
}

//=======================================================================
// Implementation Details: Index Access
//=======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void indexAccess(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    // create a Tensor filled with zeros
    NSL::Tensor<Type> T(sizes...);
    Type * T_ptr = T.data();

    // fill the Tensor with data
    for(NSL::size_t i = 0; i < numElements; ++i){
        T_ptr[i] = static_cast<Type>(i);
    }

    // check linear index access
    for(NSL::size_t i = 0; i < numElements; ++i){
        // check value match
        REQUIRE(T[i] == static_cast<Type>(i));
        // check address match
        REQUIRE(&T[i] == (T_ptr+i) );
    }

    // check dimensional index access
    // 1D
    if constexpr(sizeof...(sizes) == 1){
        for(NSL::size_t i = 0; i < shape[0]; ++i){
            // check value match
            REQUIRE(T(i) == static_cast<Type>(linearIndex(T.strides(),i)));
            // check addess match
            REQUIRE(&T(i) == (T_ptr+linearIndex(T.strides(), i)) );
        }
    } else if constexpr(sizeof...(sizes) == 2) {
        for(NSL::size_t i = 0; i < shape[0]; ++i){
        for(NSL::size_t j = 0; j < shape[1]; ++j){
            // check value match
            REQUIRE(T(i,j) == static_cast<Type>(linearIndex(T.strides(),i,j)));
            // check addess match
            REQUIRE(&T(i,j) == (T_ptr+linearIndex(T.strides(),i,j)) );
        }}
    } else if constexpr(sizeof...(sizes) == 3) {
        for(NSL::size_t i = 0; i < shape[0]; ++i){
        for(NSL::size_t j = 0; j < shape[1]; ++j){
        for(NSL::size_t k = 0; k < shape[2]; ++k){
            // check value match
            REQUIRE(T(i,j,k) == static_cast<Type>(linearIndex(T.strides(),i,j,k)));
            // check addess match
            REQUIRE(&T(i,j,k) == (T_ptr+linearIndex(T.strides(),i,j,k)) );
        }}}
    } else if constexpr(sizeof...(sizes) == 4) {
        for(NSL::size_t i = 0; i < shape[0]; ++i){
        for(NSL::size_t j = 0; j < shape[1]; ++j){
        for(NSL::size_t k = 0; k < shape[2]; ++k){
        for(NSL::size_t l = 0; l < shape[3]; ++l){
            // check value match
            REQUIRE(T(i,j,k,l) == static_cast<Type>(linearIndex(T.strides(),i,j,k,l)));
            // check addess match
            REQUIRE(&T(i,j,k,l) == (T_ptr+linearIndex(T.strides(),i,j,k,l)) );
        }}}}
    } else {
        INFO("Random Access: Index, has no implementation for dim>4");
        REQUIRE(false);
    }

}


//=======================================================================
// Implementation Details: Slice Access
//=======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void sliceAccess(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    INFO( "dim :" + std::to_string(dim) );
    INFO( "type :" + std::string(typeid(Type).name()) );

    // create a Tensor filled with zeros
    NSL::Tensor<Type> T(sizes...);
    Type * T_ptr = T.data();

    // fill the Tensor with data
    for(NSL::size_t i = 0; i < numElements; ++i){
        T_ptr[i] = static_cast<Type>(i);
    }

    // 1D
    if constexpr (sizeof...(sizes) == 1){
        for(NSL::size_t start = 0; start < shape[0]; ++start){
        for(NSL::size_t stop = start+1; stop < shape[0]; ++stop){
        for(NSL::size_t step = 1; step < stop - start; ++step){
            NSL::size_t sliceSize = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop-start)/step)); 
            NSL::Tensor<Type> Tsliced = T(NSL::Slice(start,stop,step));

            INFO( "start: " + std::to_string(start) );
            INFO( "stop : " + std::to_string(stop) );
            INFO( "step : " + std::to_string(step) );
            INFO( "size : " + std::to_string(sliceSize) );

            // check dimension
            REQUIRE( Tsliced.dim() == 1 );
            // check shape
            REQUIRE( Tsliced.shape(0) == sliceSize );
            // check number of elements
            REQUIRE( Tsliced.numel() == sliceSize );
            for(NSL::size_t i = 0; i < sliceSize; ++i){
                // check that the values match original tensor
                REQUIRE(Tsliced(i) == static_cast<Type>( start+i*step ));
                // check that the addresses match the original tensor
                REQUIRE(&Tsliced(i) == &T(start+i*step));
            }

        }}}
    // 2D
    } else if constexpr(sizeof...(sizes) == 2) { 
        for(NSL::size_t start0 = 0; start0 < shape[0]; ++start0){
        for(NSL::size_t stop0 = start0+1; stop0 < shape[0]; ++stop0){
        for(NSL::size_t step0 = 1; step0 < stop0 - start0; ++step0){

            NSL::size_t sliceSize0 = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop0-start0)/step0)); 

            INFO( "start0: " + std::to_string(start0) );
            INFO( "stop0 : " + std::to_string(stop0) );
            INFO( "step0 : " + std::to_string(step0) );
            INFO( "size0 : " + std::to_string(sliceSize0) );

            for(NSL::size_t start1 = 0; start1 < shape[1]; ++start1){
            for(NSL::size_t stop1 = start1+1; stop1 < shape[1]; ++stop1){
            for(NSL::size_t step1 = 1; step1 < stop1 - start1; ++step1){
                NSL::size_t sliceSize1 = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop1-start1)/step1)); 
                
                std::array<NSL::size_t,2> sliceSize_arr{sliceSize0,sliceSize1};

                INFO( "start1: " + std::to_string(start1) );
                INFO( "stop1 : " + std::to_string(stop1) );
                INFO( "step1 : " + std::to_string(step1) );
                INFO( "size1 : " + std::to_string(sliceSize1) );

                NSL::Tensor<Type> Tsliced = T(NSL::Slice(start0,stop0,step0), NSL::Slice(start1,stop1,step1));

                // check dimension
                REQUIRE( Tsliced.dim() == 2 );
                // check shape
                for(NSL::size_t d = 0; d < 2; ++d){
                    REQUIRE( Tsliced.shape(d) == sliceSize_arr[d] );
                }
                // check number of elements
                REQUIRE( Tsliced.numel() == sliceSize0*sliceSize1 );
                for(NSL::size_t i = 0; i < sliceSize0; ++i){
                for(NSL::size_t j = 0; j < sliceSize1; ++j){
                    NSL::size_t index = linearIndex(T.strides(), start0+i*step0,start1+j*step1);
                    // check that the values match original tensor
                    REQUIRE(Tsliced(i,j) == static_cast<Type>(index) );
                    // check that the addresses match the original tensor
                    REQUIRE(&Tsliced(i,j) == &T[index]);
                }}
            }}} // slice dim=1
        }}} // slice dim=0
    // 3D
    } else if constexpr(sizeof...(sizes) == 3) {
        for(NSL::size_t start0 = 0; start0 < shape[0]; ++start0){
        for(NSL::size_t stop0 = start0+1; stop0 < shape[0]; ++stop0){
        for(NSL::size_t step0 = 1; step0 < stop0 - start0; ++step0){

            NSL::size_t sliceSize0 = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop0-start0)/step0)); 

            INFO( "start0: " + std::to_string(start0) );
            INFO( "stop0 : " + std::to_string(stop0) );
            INFO( "step0 : " + std::to_string(step0) );
            INFO( "size0 : " + std::to_string(sliceSize0) );

            for(NSL::size_t start1 = 0; start1 < shape[1]; ++start1){
            for(NSL::size_t stop1 = start1+1; stop1 < shape[1]; ++stop1){
            for(NSL::size_t step1 = 1; step1 < stop1 - start1; ++step1){

                NSL::size_t sliceSize1 = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop1-start1)/step1)); 
                
                INFO( "start1: " + std::to_string(start1) );
                INFO( "stop1 : " + std::to_string(stop1) );
                INFO( "step1 : " + std::to_string(step1) );
                INFO( "size1 : " + std::to_string(sliceSize1) );

                for(NSL::size_t start2 = 0; start2 < shape[2]; ++start2){
                for(NSL::size_t stop2 = start2+1; stop2 < shape[2]; ++stop2){
                for(NSL::size_t step2 = 1; step2 < stop2 - start2; ++step2){

                    NSL::size_t sliceSize2 = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop2-start2)/step2)); 
                
                    INFO( "start2: " + std::to_string(start2) );
                    INFO( "stop2 : " + std::to_string(stop2) );
                    INFO( "step2 : " + std::to_string(step2) );
                    INFO( "size2 : " + std::to_string(sliceSize2) );

                    std::array<NSL::size_t,3> sliceSize_arr{sliceSize0,sliceSize1,sliceSize2};
                    
                    NSL::Tensor<Type> Tsliced = T(
                            NSL::Slice(start0,stop0,step0), 
                            NSL::Slice(start1,stop1,step1),
                            NSL::Slice(start2,stop2,step2)
                    );

                    // check dimension
                    REQUIRE( Tsliced.dim() == 3 );
                    // check shape
                    for(NSL::size_t d = 0; d < 3; ++d){
                        REQUIRE( Tsliced.shape(d) == sliceSize_arr[d] );
                    }
                    // check number of elements
                    REQUIRE( Tsliced.numel() == sliceSize0*sliceSize1*sliceSize2 );

                    for(NSL::size_t i = 0; i < sliceSize0; ++i){
                    for(NSL::size_t j = 0; j < sliceSize1; ++j){
                    for(NSL::size_t k = 0; k < sliceSize2; ++k){
                        NSL::size_t index = linearIndex(T.strides(), 
                            start0+i*step0,
                            start1+j*step1,
                            start2+k*step2
                        );

                        // check that the values match original tensor
                        REQUIRE(Tsliced(i,j,k) == static_cast<Type>(index) );
                        // check that the addresses match the original tensor
                        REQUIRE(&Tsliced(i,j,k) == &T[index]);
                    }}}

                }}} // slice dim=2
            }}} // slice dim=1
        }}} // slice dim=0

    } else {
        INFO("Random Access: Slice, has no implementation for dim>3");
        REQUIRE(false);
    }

}


// We assume that the above tests succeed, therefore, it is sufficient 
// to check if that data can be written
//=======================================================================
// Implementation Details: Index Write Access
//=======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void indexWriteAccess(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    NSL::Tensor<Type> A(sizes...);


    // linear access:
    for(NSL::size_t i = 0; i < numElements; ++i){
        A[i] = static_cast<Type>(i);
        REQUIRE(A[i] == static_cast<Type>(i));
        A[i] = static_cast<Type>(0);
    }

    if constexpr(sizeof...(sizes) == 1) {
        for(NSL::size_t i = 0; i < shape[0]; ++i){
            A(i) = static_cast<Type> (i);
            REQUIRE(A(i) == static_cast<Type>(i));
            A(i) = static_cast<Type>(0);
        }

    } else if constexpr(sizeof...(sizes) == 2){
        for(NSL::size_t i = 0; i < shape[0]; ++i){
        for(NSL::size_t j = 0; j < shape[1]; ++j){
            A(i,j) = static_cast<Type> (i+j);
            REQUIRE(A(i,j) == static_cast<Type>(i+j));
            A(i,j) = static_cast<Type>(0);
        }}
    } else if constexpr(sizeof...(sizes) == 3){
        for(NSL::size_t i = 0; i < shape[0]; ++i){
        for(NSL::size_t j = 0; j < shape[1]; ++j){
        for(NSL::size_t k = 0; k < shape[2]; ++k){
            A(i,j,k) = static_cast<Type> (i+j+k);
            REQUIRE(A(i,j,k) == static_cast<Type>(i+j+k));
            A(i,j,k) = static_cast<Type>(0);
        }}}
    } else if constexpr(sizeof...(sizes) == 4){
        for(NSL::size_t i = 0; i < shape[0]; ++i){
        for(NSL::size_t j = 0; j < shape[1]; ++j){
        for(NSL::size_t k = 0; k < shape[2]; ++k){
        for(NSL::size_t l = 0; l < shape[3]; ++l){
            A(i,j,k,l) = static_cast<Type> (i+j+k+l);
            REQUIRE(A(i,j,k,l) == static_cast<Type>(i+j+k+l));
            A(i,j,k,l) = static_cast<Type>(0);
        }}}}
    } else {
        INFO("Random Access: Slice, has no implementation for dim>4");
        REQUIRE(false);
    }

}


// We assume that the above tests succeed, therefore, it is sufficient 
// to check if that data can be written
//=======================================================================
// Implementation Details: Index Write Access
//=======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void sliceWriteAccess(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    NSL::Tensor<Type> T(sizes...);

    // 1D
    if constexpr (sizeof...(sizes) == 1){
        for(NSL::size_t start = 0; start < shape[0]; ++start){
        for(NSL::size_t stop = start+1; stop < shape[0]; ++stop){
        for(NSL::size_t step = 1; step < stop - start; ++step){
            NSL::size_t sliceSize = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop-start)/step)); 
            NSL::Tensor<Type> Tsliced = T(NSL::Slice(start,stop,step));

            INFO( "start: " + std::to_string(start) );
            INFO( "stop : " + std::to_string(stop) );
            INFO( "step : " + std::to_string(step) );
            INFO( "size : " + std::to_string(sliceSize) );


            for(NSL::size_t i = 0; i < sliceSize; ++i){
                Tsliced(i) = static_cast<Type>( start+i*step );
                // check that the values match original tensor
                REQUIRE(Tsliced(i) == static_cast<Type>( start+i*step ));
                REQUIRE(T[start+i*step] == static_cast<Type>( start+i*step ));
                Tsliced(i) = static_cast<Type>(0);
            }
        }}}
    // 2D
    } else if constexpr(sizeof...(sizes) == 2) { 
        for(NSL::size_t start0 = 0; start0 < shape[0]; ++start0){
        for(NSL::size_t stop0 = start0+1; stop0 < shape[0]; ++stop0){
        for(NSL::size_t step0 = 1; step0 < stop0 - start0; ++step0){

            NSL::size_t sliceSize0 = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop0-start0)/step0)); 

            INFO( "start0: " + std::to_string(start0) );
            INFO( "stop0 : " + std::to_string(stop0) );
            INFO( "step0 : " + std::to_string(step0) );
            INFO( "size0 : " + std::to_string(sliceSize0) );

            for(NSL::size_t start1 = 0; start1 < shape[1]; ++start1){
            for(NSL::size_t stop1 = start1+1; stop1 < shape[1]; ++stop1){
            for(NSL::size_t step1 = 1; step1 < stop1 - start1; ++step1){
                NSL::size_t sliceSize1 = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop1-start1)/step1)); 
                
                std::array<NSL::size_t,2> sliceSize_arr{sliceSize0,sliceSize1};

                INFO( "start1: " + std::to_string(start1) );
                INFO( "stop1 : " + std::to_string(stop1) );
                INFO( "step1 : " + std::to_string(step1) );
                INFO( "size1 : " + std::to_string(sliceSize1) );

                NSL::Tensor<Type> Tsliced = T(NSL::Slice(start0,stop0,step0), NSL::Slice(start1,stop1,step1));

                for(NSL::size_t i = 0; i < sliceSize0; ++i){
                for(NSL::size_t j = 0; j < sliceSize1; ++j){
                    NSL::size_t index = linearIndex(T.strides(), start0+i*step0,start1+j*step1);
                    Tsliced(i,j) = static_cast<Type>( index );
                    // check that the values match original tensor
                    REQUIRE(Tsliced(i,j) == static_cast<Type>( index ));
                    REQUIRE(T[index] == static_cast<Type>( index ));
                    Tsliced(i,j) = static_cast<Type>(0);
                }}
            }}} // slice dim=1
        }}} // slice dim=0
    // 3D
    } else if constexpr(sizeof...(sizes) == 3) {
        for(NSL::size_t start0 = 0; start0 < shape[0]; ++start0){
        for(NSL::size_t stop0 = start0+1; stop0 < shape[0]; ++stop0){
        for(NSL::size_t step0 = 1; step0 < stop0 - start0; ++step0){

            NSL::size_t sliceSize0 = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop0-start0)/step0)); 

            INFO( "start0: " + std::to_string(start0) );
            INFO( "stop0 : " + std::to_string(stop0) );
            INFO( "step0 : " + std::to_string(step0) );
            INFO( "size0 : " + std::to_string(sliceSize0) );

            for(NSL::size_t start1 = 0; start1 < shape[1]; ++start1){
            for(NSL::size_t stop1 = start1+1; stop1 < shape[1]; ++stop1){
            for(NSL::size_t step1 = 1; step1 < stop1 - start1; ++step1){

                NSL::size_t sliceSize1 = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop1-start1)/step1)); 
                
                INFO( "start1: " + std::to_string(start1) );
                INFO( "stop1 : " + std::to_string(stop1) );
                INFO( "step1 : " + std::to_string(step1) );
                INFO( "size1 : " + std::to_string(sliceSize1) );

                for(NSL::size_t start2 = 0; start2 < shape[2]; ++start2){
                for(NSL::size_t stop2 = start2+1; stop2 < shape[2]; ++stop2){
                for(NSL::size_t step2 = 1; step2 < stop2 - start2; ++step2){

                    NSL::size_t sliceSize2 = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop2-start2)/step2)); 
                
                    INFO( "start2: " + std::to_string(start2) );
                    INFO( "stop2 : " + std::to_string(stop2) );
                    INFO( "step2 : " + std::to_string(step2) );
                    INFO( "size2 : " + std::to_string(sliceSize2) );

                    std::array<NSL::size_t,3> sliceSize_arr{sliceSize0,sliceSize1,sliceSize2};
                    
                    NSL::Tensor<Type> Tsliced = T(
                            NSL::Slice(start0,stop0,step0), 
                            NSL::Slice(start1,stop1,step1),
                            NSL::Slice(start2,stop2,step2)
                    );

                    for(NSL::size_t i = 0; i < sliceSize0; ++i){
                    for(NSL::size_t j = 0; j < sliceSize1; ++j){
                    for(NSL::size_t k = 0; k < sliceSize2; ++k){
                        NSL::size_t index = linearIndex(T.strides(), 
                            start0+i*step0,
                            start1+j*step1,
                            start2+k*step2
                        );

                        Tsliced(i,j,k) = static_cast<Type>( index );
                        // check that the values match original tensor
                        REQUIRE(Tsliced(i,j,k) == static_cast<Type>( index ));
                        REQUIRE(T[index] == static_cast<Type>( index ));
                        Tsliced(i,j,k) = static_cast<Type>(0);
                    }}}
                }}} // slice dim=2
            }}} // slice dim=1
        }}} // slice dim=0

    } else {
        INFO("Random Access: Slice, has no implementation for dim>3");
        REQUIRE(false);
    }

}

//=======================================================================
// Implementation Details: Mixed Access
//=======================================================================
template<NSL::Concept::isNumber Type, NSL::Concept::isIntegral ... SizeTypes>
void mixedAccess(SizeTypes ... sizes){
    std::array<NSL::size_t,sizeof...(sizes)> shape{sizes...};
    const NSL::size_t numElements = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<NSL::size_t>());
    const NSL::size_t dim = sizeof...(sizes);

    INFO( "dim :" + std::to_string(dim) );
    INFO( "type :" + std::string(typeid(Type).name()) );

    // create a Tensor filled with zeros
    NSL::Tensor<Type> T(sizes...);
    Type * T_ptr = T.data();

    // fill the Tensor with data
    for(NSL::size_t i = 0; i < numElements; ++i){
        T_ptr[i] = static_cast<Type>(i);
    }

    // 2D
    if constexpr(sizeof...(sizes) == 2) {
        for(NSL::size_t start = 0; start < shape[0]; ++start){
        for(NSL::size_t stop = start+1; stop < shape[0]; ++stop){
        for(NSL::size_t step = 1; step < stop - start; ++step){

            NSL::size_t sliceSize = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop-start)/step)); 

            INFO( "start: " + std::to_string(start) );
            INFO( "stop : " + std::to_string(stop) );
            INFO( "step : " + std::to_string(step) );
            INFO( "size : " + std::to_string(sliceSize) );

            for(NSL::size_t index = 0; index < shape[1]; ++index){ 
                NSL::Tensor<Type> Tsliced = T(NSL::Slice(start,stop,step),index);

                // check dimension
                REQUIRE( Tsliced.dim() == 1 );
                // check shape
                REQUIRE( Tsliced.shape(0) == sliceSize);
                REQUIRE_THROWS(Tsliced.shape(1) == 0);
                // check number of elements: only the remainder of the slice
                REQUIRE( Tsliced.numel() == sliceSize);

                for(NSL::size_t i = 0; i < sliceSize; ++i){
                    NSL::size_t value = linearIndex(T.strides(), start+i*step,index);
                    // check that the values match original tensor
                    REQUIRE(Tsliced(i) == static_cast<Type>(value) );
                    // check that the addresses match the original tensor
                    REQUIRE(&Tsliced(i) == &T[value]);
                }
            } // index dim=1
        }}} // slice dim=0
    
        // and the other way round
        for(NSL::size_t start = 0; start < shape[1]; ++start){
        for(NSL::size_t stop = start+1; stop < shape[1]; ++stop){
        for(NSL::size_t step = 1; step < stop - start; ++step){

            NSL::size_t sliceSize = static_cast<NSL::size_t>(std::ceil(static_cast<float>(stop-start)/step)); 

            INFO( "start: " + std::to_string(start) );
            INFO( "stop : " + std::to_string(stop) );
            INFO( "step : " + std::to_string(step) );
            INFO( "size : " + std::to_string(sliceSize) );

            for(NSL::size_t index = 0; index < shape[0]; ++index){ 
                NSL::Tensor<Type> Tsliced = T(index,NSL::Slice(start,stop,step));

                // check dimension
                REQUIRE( Tsliced.dim() == 1 );
                // check shape
                REQUIRE( Tsliced.shape(0) == sliceSize);
                REQUIRE_THROWS(Tsliced.shape(1) == 0);
                // check number of elements: only the remainder of the slice
                REQUIRE( Tsliced.numel() == sliceSize);

                for(NSL::size_t i = 0; i < sliceSize; ++i){
                    NSL::size_t value = linearIndex(T.strides(),index, start+i*step);
                    // check that the values match original tensor
                    REQUIRE(Tsliced(i) == static_cast<Type>(value) );
                    // check that the addresses match the original tensor
                    REQUIRE(&Tsliced(i) == &T[value]);
                }
            } // index dim=1
        }}} // slice dim=0
            
    } else {
        INFO("Mixed Access: Slice, has no implementation for dim != 2");
        REQUIRE(false);
    }

}
