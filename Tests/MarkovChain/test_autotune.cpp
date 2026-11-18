#include "../test.hpp"

#include <cmath>
#include <cstdlib>
#include <yaml-cpp/yaml.h>

// Tests of the mathematics and the yml interface of NSL::MCMC::Autotune
// (src/NSL/MarkovChain/autotune.tpp). The tuner itself is tested by running the HMC drivers.

namespace AT = NSL::MCMC::Autotune;

TEST_CASE( "Autotune: erfcinv", "[Autotune]" ) {
    // reference values from scipy.special.erfcinv
    const std::vector<std::pair<double, double>> ref = {
        {1e-10, 4.572824967389486}, {0.01, 1.8213863677184499}, {0.15, 1.017902464832028},
        {0.5, 0.4769362762044699}, {1.0, 0.0}, {1.5, -0.4769362762044699}, {1.99, -1.8213863677184496}
    };
    for (const auto & [x, y] : ref) {
        INFO("x = " << x);
        REQUIRE( std::abs(AT::erfcinv(x) - y) <= 1e-12 * (1. + std::abs(y)) );
        REQUIRE( std::abs(std::erfc(AT::erfcinv(x)) - x) <= 1e-12 * x );
    }
    REQUIRE_THROWS( AT::erfcinv(0.) );
    REQUIRE_THROWS( AT::erfcinv(2.) );
}

TEST_CASE( "Autotune: bulk/spike statistics and the step scale", "[Autotune]" ) {
    const double N0 = 4.3;
    for (NSL::size_t nmd : {4, 6, 9, 15}) {
        // bulk acceptance of the model P_b(Nmd) = erfc((N0/Nmd)^2/2), no spikes
        const double Pb = std::erfc(0.5 * (N0 / nmd) * (N0 / nmd));
        AT::Stats s;
        for (int i = 0; i < 10000; ++i) { s.add(Pb, 0.1, 3.); }
        INFO("Nmd = " << nmd << ", P_b = " << Pb);
        REQUIRE( s.spikeFraction() == 0. );
        REQUIRE( std::abs(s.bulkAcceptance() - Pb) < 1e-12 );
        const auto [n0, dn0] = AT::stepScale(nmd, s);
        REQUIRE( std::abs(n0 - N0) < 1e-9 );
        REQUIRE( dn0 > 0 );
    }
    // a trajectory with dH > threshold (here e^{-dH} = e^{-4}) is a spike; e^{-dH} > 1 counts as 1
    AT::Stats s;
    s.add(std::exp(-4.), 1., 3.);
    s.add(2.5, 1., 3.);
    REQUIRE( s.n == 2 );
    REQUIRE( s.nBulk == 1 );
    REQUIRE( s.spikeFraction() == 0.5 );
    REQUIRE( s.bulkAcceptance() == 1. );
    REQUIRE( std::abs(s.acceptance() - 0.5 * (1. + std::exp(-4.))) < 1e-15 );
    // cost per independent configuration (Nmd + 1)(2/P - 1) MD steps
    REQUIRE( std::abs(s.cost(5) - 6. * (2. / s.acceptance() - 1.)) < 1e-12 );
    // too few bulk trajectories: no estimate
    AT::Stats few;
    few.add(0.9, 1., 3.);
    REQUIRE( std::isnan(AT::stepScale(5, few).first) );
}

TEST_CASE( "Autotune: the proposal reaches the target bulk acceptance", "[Autotune]" ) {
    const double target = 0.85, N0 = 3.0;
    const double nmd = N0 / std::sqrt(2. * AT::erfcinv(target));
    REQUIRE( std::abs(std::erfc(0.5 * (N0 / nmd) * (N0 / nmd)) - target) < 1e-12 );
    // pooled estimate of two Nmd with the same N0
    std::map<NSL::size_t, AT::Stats> S;
    for (NSL::size_t n : {5, 8}) {
        const double Pb = std::erfc(0.5 * (N0 / n) * (N0 / n));
        for (int i = 0; i < 2000; ++i) { S[n].add(Pb, 0.1, 3.); }
    }
    REQUIRE( std::abs(AT::pooledStepScale(S).first - N0) < 1e-9 );
}

TEST_CASE( "Autotune: escalation in the thermalization", "[Autotune]" ) {
    const double spike = std::exp(-5.);   // e^{-dH} of a rejected spike
    NSL::size_t acc = 99;
    // a decision needs `window` thermalization trajectories at the current Nmd
    AT::Record r;
    for (int i = 0; i < 49; ++i) { r.add(10, spike, false, 1., 0); }
    REQUIRE_FALSE( AT::stalled(r, 10, 50, acc) );
    r.add(10, spike, false, 1., 0);
    REQUIRE( AT::stalled(r, 10, 50, acc) );
    REQUIRE( acc == 0 );
    // after a doubling the window starts again
    REQUIRE_FALSE( AT::stalled(r, 20, 50, acc) );
    for (int i = 0; i < 50; ++i) { r.add(20, spike, false, 1., 0); }
    REQUIRE( AT::stalled(r, 20, 50, acc) );

    // at most 5%: 2 of 50 accepted is stalled, 3 of 50 is not, however long the runs of rejections
    AT::Record m;
    for (int i = 0; i < 50; ++i) { m.add(6, 0.5, i == 7 || i == 30, 1., 0); }
    REQUIRE( AT::stalled(m, 6, 50, acc) );
    REQUIRE( acc == 2 );
    m.add(6, 0.9, true, 1., 0);   // the window is now trajectories 1..50, with 3 accepted
    REQUIRE_FALSE( AT::stalled(m, 6, 50, acc) );
    REQUIRE( acc == 3 );

    // tuning trajectories are not part of the thermalization
    AT::Record t;
    for (int i = 0; i < 50; ++i) { t.add(10, spike, false, 1., 1); }
    REQUIRE_FALSE( AT::stalled(t, 10, 50, acc) );

    // window 3: three rejections in a row
    AT::Record w;
    w.add(10, spike, false, 1., 0); w.add(10, 0.9, true, 1., 0); w.add(10, spike, false, 1., 0);
    REQUIRE_FALSE( AT::stalled(w, 10, 3, acc) );
    w.add(10, spike, false, 1., 0); w.add(10, spike, false, 1., 0);
    REQUIRE( AT::stalled(w, 10, 3, acc) );

    REQUIRE( AT::Options::fromYAML(YAML::Load("HMC: {Nconf: 1}")).escalateAfter == 50 );
}

TEST_CASE( "Autotune: yml interface", "[Autotune]" ) {
    unsetenv("NSL_AUTOTUNE_PROTOCOL");
    REQUIRE( AT::modeFromYAML(YAML::Load("HMC: {Nconf: 1}")) == AT::Mode::Tune );
    REQUIRE( AT::modeFromYAML(YAML::Load("HMC: {autotune: true}")) == AT::Mode::Tune );
    REQUIRE( AT::modeFromYAML(YAML::Load("HMC: {autotune: false}")) == AT::Mode::Plain );
    REQUIRE( AT::modeFromYAML(YAML::Load("HMC: {autotune: legacy}")) == AT::Mode::Legacy );
    REQUIRE_THROWS( AT::modeFromYAML(YAML::Load("HMC: {autotune: sometimes}")) );
    setenv("NSL_AUTOTUNE_PROTOCOL", "legacy", 1);
    REQUIRE( AT::modeFromYAML(YAML::Load("HMC: {autotune: true}")) == AT::Mode::Legacy );
    unsetenv("NSL_AUTOTUNE_PROTOCOL");

    REQUIRE( AT::startDefault(true, false) == AT::Start::Cold );
    REQUIRE( AT::startDefault(true, true) == AT::Start::Cold );
    REQUIRE( AT::startDefault(false, true) == AT::Start::Cold );
    REQUIRE( AT::startDefault(false, false) == AT::Start::Gaussian );
    REQUIRE( AT::startFromYAML(YAML::Load("HMC: {Nconf: 1}"), AT::startDefault(false, false)) == AT::Start::Gaussian );
    REQUIRE( AT::startFromYAML(YAML::Load("HMC: {start: gaussian}"), AT::Start::Cold) == AT::Start::Gaussian );
    REQUIRE( AT::startFromYAML(YAML::Load("HMC: {start: cold}"), AT::Start::Gaussian) == AT::Start::Cold );
    REQUIRE_THROWS( AT::startFromYAML(YAML::Load("HMC: {start: hot}"), AT::Start::Cold) );

    const AT::Options d = AT::Options::fromYAML(YAML::Load("HMC: {Nconf: 1}"));
    REQUIRE( d.target == 0.85 );
    REQUIRE( d.minTrajectories == 50 );
    const AT::Options o = AT::Options::fromYAML(YAML::Load(
        "Autotune: {target bulk acceptance: 0.8, spike threshold: 4, batch: 10, min trajectories: 30, spike fraction: 0.2}"));
    REQUIRE( o.target == 0.8 );
    REQUIRE( o.spikeThreshold == 4. );
    REQUIRE( o.batch == 10 );
    REQUIRE( o.minTrajectories == 30 );
    REQUIRE( o.spikeFraction == 0.2 );
    REQUIRE_THROWS( AT::Options::fromYAML(YAML::Load("Autotune: {target bulk acceptance: 1.5}")) );
    REQUIRE_THROWS( AT::Options::fromYAML(YAML::Load("Autotune: {batch: 0}")) );
}
