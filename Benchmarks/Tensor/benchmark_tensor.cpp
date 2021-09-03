#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <iostream>
#include "Tensor/tensor.hpp"
#include "catch2/catch.hpp"

//#include "catch2/benchmark/catch_benchmark.hpp"


// =============================================================================
// Benchmark Cases
// =============================================================================
TEST_CASE("Benchmark 1D Tensor Constructor"){
    BENCHMARK_ADVANCED("N = 1")(Catch::Benchmark::Chronometer meter) {
//        std::vector<Catch::Benchmark::storage_for< NSL::Tensor<float> >> storage(meter.runs());
//        meter.measure([&](int i) { storage[i].construct("thing"); });
        meter.measure([](){return NSL::Tensor<float>(1);} );
    };

    BENCHMARK_ADVANCED("N = 10")(Catch::Benchmark::Chronometer meter) {
            //        std::vector<Catch::Benchmark::storage_for< NSL::Tensor<float> >> storage(meter.runs());
            //        meter.measure([&](int i) { storage[i].construct("thing"); });
            meter.measure([](){return NSL::Tensor<float>(10);} );
    };

    BENCHMARK_ADVANCED("N = 100")(Catch::Benchmark::Chronometer meter) {
            //        std::vector<Catch::Benchmark::storage_for< NSL::Tensor<float> >> storage(meter.runs());
            //        meter.measure([&](int i) { storage[i].construct("thing"); });
            meter.measure([](){return NSL::Tensor<float>(100);} );
    };

    BENCHMARK_ADVANCED("N = 1000")(Catch::Benchmark::Chronometer meter) {
            //        std::vector<Catch::Benchmark::storage_for< NSL::Tensor<float> >> storage(meter.runs());
            //        meter.measure([&](int i) { storage[i].construct("thing"); });
            meter.measure([](){return NSL::Tensor<float>(1000);} );
        };
}
//BENCHMARK_ADVANCED("destroy")(Catch::Benchmark::Chronometer meter) {
//    std::vector<Catch::Benchmark::destructable_object<std::string>> storage(meter.runs());
//    for(auto&& o : storage)
//        o.construct("thing");
//    meter.measure([&](int i) { storage[i].destruct(); });
//};