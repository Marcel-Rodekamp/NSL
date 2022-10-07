#include "../test.hpp"
#include "Configuration/Configuration.tpp"


/* The configuration is a simple unordered_map. This tests merely provide 
 * to show the interface. Besides that the configuration can be used as 
 * the unordered_map:
 * https://en.cppreference.com/w/cpp/container/unordered_map
 *
 * */

template<typename Type>
void useConfiguration();

template<typename Type>
void addPlusEqual();

template<typename Type>
void addPlus();

NSL_TEST_CASE( "Configuration", "[Configuration]" ) {
    //useConfiguration<TestType>();

    //if constexpr(!std::is_same_v<TestType,bool> and !std::is_same_v<TestType,int>){
        addPlusEqual<TestType>();
        addPlus<TestType>();
    //}
}


// ======================================================================
// Implementation
// ======================================================================

template<typename Type>
void useConfiguration(){
    // create a configuration with two fields
    NSL::Configuration<Type> config{
        {"Phi", NSL::Tensor<Type>(8,4) },
        {"Psi", NSL::Tensor<Type>(4,4) }
    };

    // access the variables and assign some values
    config["Phi"] = static_cast<Type>(1);
    config["Psi"] = static_cast<Type>(2);

    // check that the values have changed 
    REQUIRE( (config["Phi"] == static_cast<Type>(1)).all() );
    REQUIRE( (config["Psi"] == static_cast<Type>(2)).all() );

    // we can utilize the shallof copy of NSL::Tensor, to store
    // the configurations but change them at any instance
    NSL::Tensor<Type> chi(2,2);
    NSL::Configuration<Type> configExtern {
        {"chi",chi}
    };
    // Notice, 
    // config["chi"] = chi
    // performs a deepcopy hence the change would be only within the 
    // configuration but not within the instance chi.

    // change in config
    configExtern["chi"] = static_cast<Type>(3);

    // check that the values have changed
    REQUIRE( (configExtern["chi"] == static_cast<Type>(3)).all() );
    REQUIRE( (chi == static_cast<Type>(3)).all() );
    
    // change in original instance
    chi = static_cast<Type>(4);

    // check that the values have changed
    REQUIRE( (configExtern["chi"] == static_cast<Type>(4)).all() );
    REQUIRE( (chi == static_cast<Type>(4)).all() );
}


template<typename Type>
void addPlusEqual(){
    // create a configuration with two fields
    NSL::Configuration<Type> config1{
        {"Phi", NSL::Tensor<Type>(8,4) },
        {"Psi", NSL::Tensor<Type>(4,4) }
    };
    NSL::Configuration<Type> config2{
        {"Phi", NSL::Tensor<Type>(8,4) },
        {"Psi", NSL::Tensor<Type>(4,4) },
        {"Chi", NSL::Tensor<Type>(4,4) }
    };

    // fill the fields with random numbers
    if constexpr ( std::is_same_v<Type,bool> ) {
        for(auto & [key,field]: config1){
            field.randint(1);
        }
        for(auto & [key,field]: config2){
            field.randint(1);
        }
    } else if constexpr(std::is_same_v<Type,int>){
        for(auto & [key,field]: config1){
            field.randint(1000);
        }
        for(auto & [key,field]: config2){
            field.randint(1000);
        }
    } else {
        for(auto & [key,field]: config1){
            field.rand();
        }
        for(auto & [key,field]: config2){
            field.rand();
        }
    }

    // backup the first configuration
    NSL::Configuration<Type> config3(config1,true);
    
    config1 += config2;

    INFO("\n\nconfig1\n" << config1);
    INFO("\n\nconfig2\n" << config2);
    INFO("\n\nconfig3\n" << config3);

    for( auto & key: {"Phi","Psi"} ){
        INFO(key);
        // Ceck that the fields which were apperent in in config1 are 
        // added correctly
        REQUIRE((
            config1[key] == (config2[key]+config3[key])
        ).all());
        // Check that the data locality is different
        REQUIRE(config1[key].data() != config2[key].data());
        REQUIRE(config1[key].data() != config3[key].data());
        REQUIRE(config2[key].data() != config3[key].data());
    }
    // Check that the new field is copied into config1
    REQUIRE((
        config1["Chi"] == (config2["Chi"])
    ).all());
    // Check that the data locality is different
    REQUIRE(config1["Chi"].data() != config2["Chi"].data());
}

template<typename Type>
void addPlus(){
    // create a configuration with two fields
    NSL::Configuration<Type> config1{
        {"Phi", NSL::Tensor<Type>(8,4) },
        {"Psi", NSL::Tensor<Type>(4,4) }
    };
    NSL::Configuration<Type> config2{
        {"Phi", NSL::Tensor<Type>(8,4) },
        {"Psi", NSL::Tensor<Type>(4,4) },
        {"Chi", NSL::Tensor<Type>(4,4) }
    };

    // fill the fields with random numbers
    if constexpr ( std::is_same_v<Type,bool> ) {
        for(auto & [key,field]: config1){
            field.randint(1);
        }
        for(auto & [key,field]: config2){
            field.randint(1);
        }
    } else if constexpr(std::is_same_v<Type,int>){
        for(auto & [key,field]: config1){
            field.randint(1000);
        }
        for(auto & [key,field]: config2){
            field.randint(1000);
        }
    } else {
        for(auto & [key,field]: config1){
            field.rand();
        }
        for(auto & [key,field]: config2){
            field.rand();
        }
    }

    auto config3 = config1 + config2;

    INFO("\n\nconfig1\n" << config1);
    INFO("\n\nconfig2\n" << config2);
    INFO("\n\nconfig3\n" << config3);

    for( auto & key: {"Phi","Psi"} ){
        INFO(key);
        // Ceck that the fields which were apperent in in config1 are 
        // added correctly
        REQUIRE((
            config3[key] == (config1[key]+config2[key])
        ).all());
        // Check that the data locality is different
        REQUIRE(config1[key].data() != config2[key].data());
        REQUIRE(config1[key].data() != config3[key].data());
        REQUIRE(config2[key].data() != config3[key].data());
    }
    // Check that the new field is copied into config1
    REQUIRE((
        config3["Chi"] == (config2["Chi"])
    ).all());
    // Check that the data locality is different
    REQUIRE(config3["Chi"].data() != config2["Chi"].data());
}
