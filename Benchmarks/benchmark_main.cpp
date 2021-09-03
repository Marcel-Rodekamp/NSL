#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "catch2/catch.hpp"
#include <Kokkos_Core.hpp>

int main( int argc, char* argv[] ) {
    // ToDo: We should hide Kokkos::initialization in a NSL::initialization()!
    Kokkos::initialize(argc, argv);

    int result = Catch::Session().run( argc, argv );

    Kokkos::finalize();
    return result;
}