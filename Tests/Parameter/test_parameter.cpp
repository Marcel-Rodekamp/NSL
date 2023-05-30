#include "../test.hpp"

NSL_TEST_CASE("Parameter", "[Parameter]"){
    NSL::Parameter params;
    params.addParameter<TestType>("myParam1");
    params.addParameter<std::string>("myParam2");

    // Params can be converted to its correct Type 
    REQUIRE(params["myParam1"].to<TestType>() == TestType{});
    REQUIRE(params["myParam2"].to<std::string>() == std::string());

    // Params can not be converted to other Type
    REQUIRE_THROWS(params["myParam1"].to<std::string>());
    REQUIRE_THROWS(params["myParam2"].to<TestType>());

    // explicit conversion, if this works implicit conversion works automatically
    REQUIRE_NOTHROW(TestType(params["myParam1"]));
    REQUIRE_NOTHROW(std::string(params["myParam2"]));

    // Params can not be converted to other Type
    REQUIRE_THROWS(std::string(params["myParam1"]));
    REQUIRE_THROWS(TestType(params["myParam2"]));

    // assignment with the correct type should work
    REQUIRE_NOTHROW(params["myParam1"] = static_cast<TestType>(2));
    // assignment with the wrong type should fail
    REQUIRE_THROWS(params["myParam1"] = std::string("Test"));

}
