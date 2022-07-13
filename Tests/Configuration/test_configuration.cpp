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

NSL_TEST_CASE( "Configuration", "[Configuration]" ) {
    useConfiguration<TestType>();
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

