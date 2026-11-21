#include "../test.hpp"

// NSL::MCMC::BridgeLink (src/NSL/MarkovChain/bridgeLink.tpp) on a 4-site ring (bipartite, mu = 0):
//  - the bridge ratios R_k from the force equal the determinant ratios exp(-(S_F(phi + c_k) - S_F(phi)));
//  - Lambda = mean_k R_k >= cos^2(theta/2);
//  - the heat bath installs the shift in every copy of the fermion action (also inside NSL::Action::Action);
//  - the heat bath draws k with probability R_k / sum R.

TEST_CASE("BridgeLink: ratios, bound, shared shift and heat bath", "[MarkovChain,bridge]") {
    typedef NSL::complex<double> Type;
    const NSL::size_t Nt = 8, Nx = 4;
    NSL::Lattice::Ring<Type> lattice(Nx);
    NSL::Parameter params;
    params["beta"] = Type(2.);
    params["U"] = Type(4.);
    params["mu"] = 0.;
    params["Nt"] = Nt;

    using FermionAction = NSL::Action::HubbardFermionAction<Type, decltype(lattice),
                                                             NSL::FermionMatrix::HubbardExp<Type, decltype(lattice)>>;
    NSL::Action::HubbardGaugeAction<Type> S_gauge(params);
    FermionAction S_fermion(lattice, params);
    FermionAction unshifted(lattice, params);

    const double theta = M_PI / 2;
    const NSL::size_t slice = 3;
    NSL::MCMC::BridgeLink<Type, FermionAction> bridge(S_fermion, Nt, Nx, theta, slice);
    NSL::Action::Action S = S_gauge + S_fermion;                 // a copy that shares the shift

    auto shifted = [&](const NSL::Tensor<Type> & phi, NSL::size_t k) {
        NSL::Tensor<Type> p(phi, true);
        p(slice, k / 2) += Type(theta * ((k % 2) ? -1. : 1.));
        return p;
    };

    for (int sample = 0; sample < 4; ++sample) {
        NSL::Tensor<Type> phi(Nt, Nx);
        phi.randn();
        phi.imag() = 0;
        phi *= std::sqrt(2. / Nt * 4.);                           // the scale sqrt(delta U) of the gauge action

        const std::vector<double> R = bridge.ratios(phi);
        const double S0 = NSL::real(unshifted.eval(phi));
        double mean = 0;
        for (NSL::size_t k = 0; k < R.size(); ++k) {
            const double ratio = std::exp(-(NSL::real(unshifted.eval(shifted(phi, k))) - S0));
            INFO("sample " << sample << ", k = " << k << ": R = " << R[k] << ", determinant ratio = " << ratio);
            REQUIRE(std::abs(R[k] - ratio) <= 1e-8 * ratio);
            mean += R[k] / R.size();
        }
        REQUIRE(mean >= std::cos(theta / 2) * std::cos(theta / 2) * (1 - 1e-12));

        NSL::Configuration<Type> config{{"phi", phi}};
        REQUIRE(bridge(config));
        const NSL::size_t k = bridge.term();
        const double Sk = NSL::real(unshifted.eval(shifted(phi, k)));
        REQUIRE(std::abs(NSL::real(S_fermion.eval(phi)) - Sk) <= 1e-10 * std::abs(Sk));
        REQUIRE(std::abs(NSL::real(S(config)) - (NSL::real(S_gauge.eval(phi)) + Sk)) <= 1e-10 * std::abs(Sk));
        REQUIRE(std::abs(bridge.lastLambda() - mean) <= 1e-12 * mean);
    }

    // heat-bath frequencies at one configuration
    NSL::Tensor<Type> phi(Nt, Nx);
    phi.randn();
    phi.imag() = 0;
    phi *= std::sqrt(2. / Nt * 4.);
    const std::vector<double> R = bridge.ratios(phi);
    double total = 0;
    for (double r : R) total += r;
    const int draws = 4000;
    std::vector<int> count(R.size(), 0);
    NSL::Configuration<Type> config{{"phi", phi}};
    for (int d = 0; d < draws; ++d) {
        bridge(config);
        ++count[bridge.term()];
    }
    for (NSL::size_t k = 0; k < R.size(); ++k) {
        const double p = R[k] / total, sigma = std::sqrt(draws * p * (1 - p));
        INFO("k = " << k << ": " << count[k] << " draws, expected " << draws * p);
        REQUIRE(std::abs(count[k] - draws * p) <= 5 * sigma + 1);
    }
}
