#include "../test.hpp"

NSL_TEST_CASE("Parameter", "[Parameter]"){
    NSL::Parameter params;

    // let us read in a bool flag
    REQUIRE_NOTHROW(params["key1"] = true);
    // let us read in a double parameter
    REQUIRE_NOTHROW(params["key2"] = 1.235);
    // let us, finally, read in a int parameter
    REQUIRE_NOTHROW(params["key3"] = 2);

    // sometimes you have a parameter that is computed out of two parameters:
    REQUIRE_NOTHROW(params["key4"] = params["key3"]*params["key2"]);

    // alternatively you can compute parameters with non-GenType numbers as long as the underlying
    // types can be computed
    REQUIRE_NOTHROW(params["key5"] = 0.2*params["key3"]);

    // reassign with a different type
    REQUIRE_NOTHROW(params["key2"] = std::string("Test"));

}
