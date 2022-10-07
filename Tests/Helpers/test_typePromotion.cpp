#include "../test.hpp"
#include "IO/to_string.tpp"
#include "LinAlg/inner_product.tpp"
#include "LinAlg/mat_vec.tpp"
#include "typePromotion.hpp"

// Even though much of the type conversion belongs to either Tensor or LinAlg
// I put everything in here in case we need to change the types. Then we have
// all these explicitly coded types in one place.

void typePromotion();

void typePromotionTensor();

template<NSL::Concept::isNumber TypeLeft, NSL::Concept::isNumber TypeRight>
void typeConversionCopyConstructTensor();

template<NSL::Concept::isNumber Type1, NSL::Concept::isNumber Type2>
void typeConversionTensorOperators();

// todo;
void typeConversionLinAlg();

TEST_CASE("Type Promotion", "[TypePromotion]"){
    typePromotion();
}

TEST_CASE("Tensor Type Promotion", "[Tensor,Type, Type Promotion]"){
    typePromotionTensor();
}

NSL_TEST_CASE("Tensor Copy Construct Type Conversion", "[Copy Constructor, Tensor, Type, Type Conversion]"){
    typeConversionCopyConstructTensor<TestType,bool>();
    typeConversionCopyConstructTensor<TestType,int>();
    typeConversionCopyConstructTensor<TestType,float>();
    typeConversionCopyConstructTensor<TestType,double>();
    typeConversionCopyConstructTensor<TestType,NSL::complex<float>>();
    typeConversionCopyConstructTensor<TestType,NSL::complex<double>>();
}

NSL_TEST_CASE("Tensor Operator Type Conversion", "[Operator, Tensor, Type, Type Conversion]"){
    if (!std::is_same_v<TestType,bool>){
        typeConversionTensorOperators<TestType, int>();
        typeConversionTensorOperators<TestType, float>();
        typeConversionTensorOperators<TestType, double>();
        typeConversionTensorOperators<TestType, NSL::complex<float>>();
        typeConversionTensorOperators<TestType, NSL::complex<double>>();
    }
}

void typePromotion(){
    REQUIRE( std::is_same_v<bool                ,NSL::CommonTypeOf<bool,bool>> );
    REQUIRE( std::is_same_v<int                 ,NSL::CommonTypeOf<bool,int>> );
    REQUIRE( std::is_same_v<float               ,NSL::CommonTypeOf<bool,float>> );
    REQUIRE( std::is_same_v<double              ,NSL::CommonTypeOf<bool,double>> );
    REQUIRE( std::is_same_v<NSL::complex<float> ,NSL::CommonTypeOf<bool,NSL::complex<float>>> );
    REQUIRE( std::is_same_v<NSL::complex<double>,NSL::CommonTypeOf<bool,NSL::complex<double>>> );

    REQUIRE( std::is_same_v<int                 ,NSL::CommonTypeOf<int,bool>> );
    REQUIRE( std::is_same_v<int                 ,NSL::CommonTypeOf<int,int>> );
    REQUIRE( std::is_same_v<float               ,NSL::CommonTypeOf<int,float>> );
    REQUIRE( std::is_same_v<double              ,NSL::CommonTypeOf<int,double>> );
    REQUIRE( std::is_same_v<NSL::complex<float> ,NSL::CommonTypeOf<int,NSL::complex<float>>> );
    REQUIRE( std::is_same_v<NSL::complex<double>,NSL::CommonTypeOf<int,NSL::complex<double>>> );

    REQUIRE( std::is_same_v<float               ,NSL::CommonTypeOf<float,bool>> );
    REQUIRE( std::is_same_v<float               ,NSL::CommonTypeOf<float,int>> );
    REQUIRE( std::is_same_v<float               ,NSL::CommonTypeOf<float,float>> );
    REQUIRE( std::is_same_v<double              ,NSL::CommonTypeOf<float,double>> );
    REQUIRE( std::is_same_v<NSL::complex<float> ,NSL::CommonTypeOf<float,NSL::complex<float>>> );
    REQUIRE( std::is_same_v<NSL::complex<double>,NSL::CommonTypeOf<float,NSL::complex<double>>> );

    REQUIRE( std::is_same_v<double              ,NSL::CommonTypeOf<double,bool>> );
    REQUIRE( std::is_same_v<double              ,NSL::CommonTypeOf<double,int>> );
    REQUIRE( std::is_same_v<double              ,NSL::CommonTypeOf<double,float>> );
    REQUIRE( std::is_same_v<double              ,NSL::CommonTypeOf<double,double>> );
    REQUIRE( std::is_same_v<NSL::complex<double>,NSL::CommonTypeOf<double,NSL::complex<float>>> );
    REQUIRE( std::is_same_v<NSL::complex<double>,NSL::CommonTypeOf<double,NSL::complex<double>>> );

    REQUIRE( std::is_same_v<NSL::complex<float> ,NSL::CommonTypeOf<NSL::complex<float>,bool>> );
    REQUIRE( std::is_same_v<NSL::complex<float> ,NSL::CommonTypeOf<NSL::complex<float>,int>> );
    REQUIRE( std::is_same_v<NSL::complex<float> ,NSL::CommonTypeOf<NSL::complex<float>,float>> );
    REQUIRE( std::is_same_v<NSL::complex<double>,NSL::CommonTypeOf<NSL::complex<float>,double>> );
    REQUIRE( std::is_same_v<NSL::complex<float> ,NSL::CommonTypeOf<NSL::complex<float>,NSL::complex<float>>> );
    REQUIRE( std::is_same_v<NSL::complex<double>,NSL::CommonTypeOf<NSL::complex<float>,NSL::complex<double>>> );

    REQUIRE( std::is_same_v<NSL::complex<double>,NSL::CommonTypeOf<NSL::complex<double>,bool>> );
    REQUIRE( std::is_same_v<NSL::complex<double>,NSL::CommonTypeOf<NSL::complex<double>,int>> );
    REQUIRE( std::is_same_v<NSL::complex<double>,NSL::CommonTypeOf<NSL::complex<double>,float>> );
    REQUIRE( std::is_same_v<NSL::complex<double>,NSL::CommonTypeOf<NSL::complex<double>,double>> );
    REQUIRE( std::is_same_v<NSL::complex<double>,NSL::CommonTypeOf<NSL::complex<double>,NSL::complex<float>>> );
    REQUIRE( std::is_same_v<NSL::complex<double>,NSL::CommonTypeOf<NSL::complex<double>,NSL::complex<double>>> );
}

void typePromotionTensor(){
    REQUIRE( std::is_same_v<NSL::Tensor<bool>                ,NSL::CommonTypeOf<NSL::Tensor<bool>,NSL::Tensor<bool>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<int>                 ,NSL::CommonTypeOf<NSL::Tensor<bool>,NSL::Tensor<int>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<float>               ,NSL::CommonTypeOf<NSL::Tensor<bool>,NSL::Tensor<float>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<double>              ,NSL::CommonTypeOf<NSL::Tensor<bool>,NSL::Tensor<double>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<float>> ,NSL::CommonTypeOf<NSL::Tensor<bool>,NSL::Tensor<NSL::complex<float>>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<double>>,NSL::CommonTypeOf<NSL::Tensor<bool>,NSL::Tensor<NSL::complex<double>>>> );

    REQUIRE( std::is_same_v<NSL::Tensor<int>                 ,NSL::CommonTypeOf<NSL::Tensor<int>,NSL::Tensor<bool>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<int>                 ,NSL::CommonTypeOf<NSL::Tensor<int>,NSL::Tensor<int>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<float>               ,NSL::CommonTypeOf<NSL::Tensor<int>,NSL::Tensor<float>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<double>              ,NSL::CommonTypeOf<NSL::Tensor<int>,NSL::Tensor<double>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<float>> ,NSL::CommonTypeOf<NSL::Tensor<int>,NSL::Tensor<NSL::complex<float>>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<double>>,NSL::CommonTypeOf<NSL::Tensor<int>,NSL::Tensor<NSL::complex<double>>>> );

    REQUIRE( std::is_same_v<NSL::Tensor<float>               ,NSL::CommonTypeOf<NSL::Tensor<float>,NSL::Tensor<bool>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<float>               ,NSL::CommonTypeOf<NSL::Tensor<float>,NSL::Tensor<int>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<float>               ,NSL::CommonTypeOf<NSL::Tensor<float>,NSL::Tensor<float>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<double>              ,NSL::CommonTypeOf<NSL::Tensor<float>,NSL::Tensor<double>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<float>> ,NSL::CommonTypeOf<NSL::Tensor<float>,NSL::Tensor<NSL::complex<float>>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<double>>,NSL::CommonTypeOf<NSL::Tensor<float>,NSL::Tensor<NSL::complex<double>>>> );

    REQUIRE( std::is_same_v<NSL::Tensor<double>              ,NSL::CommonTypeOf<NSL::Tensor<double>,NSL::Tensor<bool>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<double>              ,NSL::CommonTypeOf<NSL::Tensor<double>,NSL::Tensor<int>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<double>              ,NSL::CommonTypeOf<NSL::Tensor<double>,NSL::Tensor<float>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<double>              ,NSL::CommonTypeOf<NSL::Tensor<double>,NSL::Tensor<double>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<double>>,NSL::CommonTypeOf<NSL::Tensor<double>,NSL::Tensor<NSL::complex<float>>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<double>>,NSL::CommonTypeOf<NSL::Tensor<double>,NSL::Tensor<NSL::complex<double>>>> );

    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<float>> ,NSL::CommonTypeOf<NSL::Tensor<NSL::complex<float>>,NSL::Tensor<bool>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<float>> ,NSL::CommonTypeOf<NSL::Tensor<NSL::complex<float>>,NSL::Tensor<int>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<float>> ,NSL::CommonTypeOf<NSL::Tensor<NSL::complex<float>>,NSL::Tensor<float>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<double>>,NSL::CommonTypeOf<NSL::Tensor<NSL::complex<float>>,NSL::Tensor<double>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<float>> ,NSL::CommonTypeOf<NSL::Tensor<NSL::complex<float>>,NSL::Tensor<NSL::complex<float>>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<double>>,NSL::CommonTypeOf<NSL::Tensor<NSL::complex<float>>,NSL::Tensor<NSL::complex<double>>>> );

    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<double>>,NSL::CommonTypeOf<NSL::Tensor<NSL::complex<double>>,NSL::Tensor<bool>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<double>>,NSL::CommonTypeOf<NSL::Tensor<NSL::complex<double>>,NSL::Tensor<int>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<double>>,NSL::CommonTypeOf<NSL::Tensor<NSL::complex<double>>,NSL::Tensor<float>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<double>>,NSL::CommonTypeOf<NSL::Tensor<NSL::complex<double>>,NSL::Tensor<double>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<double>>,NSL::CommonTypeOf<NSL::Tensor<NSL::complex<double>>,NSL::Tensor<NSL::complex<float>>>> );
    REQUIRE( std::is_same_v<NSL::Tensor<NSL::complex<double>>,NSL::CommonTypeOf<NSL::Tensor<NSL::complex<double>>,NSL::Tensor<NSL::complex<double>>>> );
}

template<NSL::Concept::isNumber TypeLeft, NSL::Concept::isNumber TypeRight>
void typeConversionCopyConstructTensor(){
    // check the normal copy construct
    NSL::Tensor<TypeRight> T(1);
    // this must always be a deep copy as the type of the underlying data changes
    NSL::Tensor<TypeLeft> Tcopy(T);
    // check that the NSL::Tensor has the expected type
    REQUIRE( std::is_same_v<NSL::Tensor<TypeLeft>, decltype(Tcopy)> );
    // check that the underlying torch::Tensor has the expected type
    REQUIRE( torch::Tensor(NSL::Tensor<TypeLeft>(1)).dtype() == torch::Tensor(Tcopy).dtype() );
    REQUIRE( torch::Tensor(NSL::Tensor<TypeRight>(1)).dtype() == torch::Tensor(T).dtype() );

    // Now the same with assignment construct
    NSL::Tensor<TypeRight> t(1);
    // this must always be a deep copy as the type of the underlying data changes
    NSL::Tensor<TypeLeft> tcopy = t;
    // check that the NSL::Tensor has the expected type
    REQUIRE( std::is_same_v<NSL::Tensor<TypeLeft>, decltype(tcopy)> );
    // check that the underlying torch::Tensor has the expected type
    REQUIRE( torch::Tensor(NSL::Tensor<TypeLeft>(1)).dtype() == torch::Tensor(tcopy).dtype() );
    REQUIRE( torch::Tensor(NSL::Tensor<TypeRight>(1)).dtype() == torch::Tensor(t).dtype() );
}

template<NSL::Concept::isNumber Type1, NSL::Concept::isNumber Type2>
class Tester {
    public:
    Tester() = default;
    Tester(const Tester<Type1,Type2> &) = default;
    Tester(Tester<Type1,Type2> &&) = default;

    template<NSL::Concept::isNumber ReturnType = NSL::CommonTypeOf<Type1,Type2> >
    void test(auto callable){
        NSL::Tensor<Type1> T1(1);
        T1[0] = static_cast<Type1>(1); 
        
        NSL::Tensor<Type2> T2(1);
        T2[0] = static_cast<Type2>(1); 

        Type1 V1 = static_cast<Type1>(1);
        Type2 V2 = static_cast<Type2>(1);

        this->testConversion<NSL::Tensor<ReturnType>>( callable(T1,T2) ); 
        this->testConversion<NSL::Tensor<ReturnType>>( callable(T2,T1) ); 
                                                   
        this->testConversion<NSL::Tensor<ReturnType>>( callable(V1,T2) );
        this->testConversion<NSL::Tensor<ReturnType>>( callable(T2,V1) );
                                                   
        this->testConversion<NSL::Tensor<ReturnType>>( callable(V2,T1) );
        this->testConversion<NSL::Tensor<ReturnType>>( callable(T1,V2) );

    }

    template<NSL::Concept::isNumber ReturnType = NSL::CommonTypeOf<Type1,Type2> >
    void testOpEq(auto callable){
        NSL::Tensor<Type1> T1(1);
        T1[0] = static_cast<Type1>(1); 
        
        NSL::Tensor<Type2> T2(1);
        T2[0] = static_cast<Type2>(1); 

        Type1 V1 = static_cast<Type1>(1);
        Type2 V2 = static_cast<Type2>(1);

        this->testConversion<NSL::Tensor<ReturnType>>( callable(T1,T2) ); 
        this->testConversion<NSL::Tensor<ReturnType>>( callable(T2,T1) ); 
                                                   
        this->testConversion<NSL::Tensor<ReturnType>>( callable(T2,V1) );
        this->testConversion<NSL::Tensor<ReturnType>>( callable(T1,V2) );

    }

    void testLinAlg(){

        NSL::Tensor<Type1> mat1(2,2);
        NSL::Tensor<Type2> mat2(2,2);
        NSL::Tensor<Type1> vec1(2);
        NSL::Tensor<Type2> vec2(2);
        
        this->testConversion<NSL::Tensor<NSL::CommonTypeOf<Type1,Type2>>>(
            NSL::LinAlg::mat_vec(mat1,vec2)
        );
        this->testConversion<NSL::Tensor<NSL::CommonTypeOf<Type1,Type2>>>(
            NSL::LinAlg::mat_vec(mat2,vec1)
        );

        this->testConversion<NSL::Tensor<NSL::CommonTypeOf<Type1,Type2>>>(
            NSL::LinAlg::mat_mul(mat1,vec2)
        );
        this->testConversion<NSL::Tensor<NSL::CommonTypeOf<Type1,Type2>>>(
            NSL::LinAlg::mat_mul(mat2,vec1)
        );

        this->testConversionScalarReturn<NSL::CommonTypeOf<Type1,Type2>>(
            NSL::LinAlg::inner_product(vec1,vec2)
        );
    }

    template<typename T, typename U>
    void testConversion(U result){
        INFO(std::string("T type: ") + typeid(T).name());
        INFO(std::string("U type: ") + typeid(U).name());
        INFO(torch::Tensor(result).dtype());
        REQUIRE(std::is_same_v<T,U>);
    }

    template<typename T, typename U>
    void testConversionScalarReturn(U){
        INFO(std::string("T type: ") + typeid(T).name());
        INFO(std::string("U type: ") + typeid(U).name());
        REQUIRE(std::is_same_v<T,U>);
    }
};

template<NSL::Concept::isNumber Type1, NSL::Concept::isNumber Type2>
void typeConversionTensorOperators(){
    Tester<Type1,Type2> tester;

    tester.test([](auto T1, auto T2){return T1+T2;});
    tester.test([](auto T1, auto T2){return T1-T2;});
    tester.test([](auto T1, auto T2){return T1*T2;});
    // devision comes with atleast float 
    tester.template test<NSL::CommonTypeOf<NSL::CommonTypeOf<Type1,Type2>, float>>(
        [](auto T1, auto T2){return T1/T2;}
    );

    tester.testOpEq([](auto T1, auto T2){return T1+=T2;});
    tester.testOpEq([](auto T1, auto T2){return T1-=T2;});
    tester.testOpEq([](auto T1, auto T2){return T1*=T2;});
    // devision comes with atleast float 
    tester.template testOpEq<NSL::CommonTypeOf<NSL::CommonTypeOf<Type1,Type2>, float>>(
        [](auto T1, auto T2){return T1/=T2;}
    );
    
    tester.testLinAlg();

}

