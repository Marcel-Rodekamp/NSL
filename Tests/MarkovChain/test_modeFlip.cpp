#include "../test.hpp"
#include <filesystem>

// NSL::MCMC::ModeFlip (src/NSL/MarkovChain/modeFlip.tpp) on a 4-site ring with a long-range interaction V:
//  - interactionModes: orthonormal eigenvectors of V (the sites for V = U 1); non-orthonormal modes are refused;
//  - the reflection is an involution, keeps the imaginary part and leaves the PPP gauge action unchanged;
//  - the Metropolis update accepts the flip of mode k with probability min(1, e^{-dS_k}), dS_k from the full action
//    (gauge + charge-basis fermion action, with a bridge shift), and returns the action value of its state;
//  - with 0 proposals it returns the state and draws no random numbers;
//  - in RadialHMC the extra update runs once per combined step.

namespace {

typedef NSL::complex<double> Type;
const NSL::size_t Nt = 8, Nx = 4;

// V on the 4-ring: U on site, U / (1 + d) at ring distance d (positive definite; eigenvalues 28/3, 8/3, 8/3, 4/3)
NSL::Tensor<double> ringV(double U) {
    NSL::Tensor<double> V(Nx, Nx);
    for (NSL::size_t x = 0; x < Nx; ++x) {
        for (NSL::size_t y = 0; y < Nx; ++y) {
            const NSL::size_t d = std::min<NSL::size_t>((x + Nx - y) % Nx, (y + Nx - x) % Nx);
            V(x, y) = U / (1. + double(d));
        }
    }
    return V;
}

NSL::Parameter ringParams() {
    NSL::Parameter params;
    params["beta"] = Type(2.);
    params["U"] = Type(4.);
    params["mu"] = 0.;
    params["Nt"] = Nt;
    params["Nx"] = Nx;
    return params;
}

NSL::Tensor<Type> gaussianField(double scale) {
    NSL::Tensor<Type> phi(Nt, Nx);
    phi.randn();
    phi.imag() = 0;
    phi *= scale;
    return phi;
}

} // namespace

TEST_CASE("ModeFlip: modes, reflection, gauge invariance", "[MarkovChain,modeflip]") {
    NSL::Parameter params = ringParams();
    const NSL::Tensor<double> V = ringV(4.);
    NSL::Action::PPPGaugeAction<Type> S_gauge(V, params);

    const NSL::Tensor<double> modes = NSL::MCMC::interactionModes(V);
    const torch::Tensor Q = torch::Tensor(modes);
    REQUIRE((torch::matmul(Q.transpose(0, 1), Q) - torch::eye(Nx, torch::kDouble)).abs().max().item<double>() < 1e-12);
    // columns are eigenvectors, largest eigenvalue first
    const torch::Tensor VQ = torch::matmul(torch::Tensor(V), Q);
    const torch::Tensor lam = (Q * VQ).sum(0);
    REQUIRE((VQ - Q * lam).abs().max().item<double>() < 1e-12);
    REQUIRE(lam[0].item<double>() >= lam[Nx - 1].item<double>());

    // V = U 1: the sites
    NSL::Tensor<double> VU(Nx, Nx);
    for (NSL::size_t x = 0; x < Nx; ++x) { VU(x, x) = 4.; }
    REQUIRE(torch::equal(torch::Tensor(NSL::MCMC::interactionModes(VU)), torch::eye(Nx, torch::kDouble)));

    NSL::Tensor<double> bad(Nx, Nx);
    bad = 1.;
    REQUIRE_THROWS_AS(NSL::MCMC::ModeFlip<Type>(bad, 1), std::invalid_argument);

    NSL::MCMC::ModeFlip<Type> flip(modes, 1);
    for (int sample = 0; sample < 3; ++sample) {
        NSL::Tensor<Type> phi = gaussianField(1.0);
        phi.imag() = 0.3;                                       // an offset: kept by the reflection
        for (NSL::size_t k = 0; k < Nx; ++k) {
            const NSL::Tensor<Type> r = flip.reflect(phi, k);
            const torch::Tensor rr = torch::Tensor(r);
            REQUIRE((torch::imag(rr) - 0.3).abs().max().item<double>() < 1e-15);
            REQUIRE((torch::Tensor(flip.reflect(r, k)) - torch::Tensor(phi)).abs().max().item<double>() < 1e-12);
            // the amplitude of mode k is reversed, the others are kept
            const torch::Tensor a0 = torch::matmul(torch::real(torch::Tensor(phi)), Q);
            const torch::Tensor a1 = torch::matmul(torch::real(rr), Q);
            REQUIRE((a1.select(1, k) + a0.select(1, k)).abs().max().item<double>() < 1e-12);
            for (NSL::size_t j = 0; j < Nx; ++j) {
                if (j != k) { REQUIRE((a1.select(1, j) - a0.select(1, j)).abs().max().item<double>() < 1e-12); }
            }
            NSL::Tensor<Type> p0(phi, true), p1(r, true);
            p0.imag() = 0;
            p1.imag() = 0;
            const double Sg0 = NSL::real(S_gauge.eval(p0)), Sg1 = NSL::real(S_gauge.eval(p1));
            INFO("sample " << sample << ", mode " << k << ": S_g " << Sg0 << " -> " << Sg1);
            REQUIRE(std::abs(Sg1 - Sg0) <= 1e-12 * std::abs(Sg0));
        }
    }
}

TEST_CASE("ModeFlip: Metropolis acceptance with the full (shifted) action", "[MarkovChain,modeflip]") {
    NSL::Parameter params = ringParams();
    NSL::Lattice::Ring<Type> lattice(Nx);
    const NSL::Tensor<double> V = ringV(4.);
    using FermionAction = NSL::Action::HubbardFermionAction<Type, decltype(lattice),
                                                             NSL::FermionMatrix::HubbardExp<Type, decltype(lattice)>>;
    NSL::Action::PPPGaugeAction<Type> S_gauge(V, params);
    FermionAction S_fermion(lattice, params);
    NSL::MCMC::BridgeLink<Type, FermionAction> bridge(S_fermion, Nt, Nx, M_PI / 2, NSL::MCMC::BridgeLink<Type, FermionAction>::AllSlices);
    NSL::Action::Action S = S_gauge + S_fermion;                 // shares the bridge shift

    NSL::MCMC::ModeFlip<Type> flip(NSL::MCMC::interactionModes(V), 1);
    torch::manual_seed(11);
    NSL::Tensor<Type> phi = gaussianField(std::sqrt(2. / Nt * 4.));
    NSL::Configuration<Type> config{{"phi", phi}};
    bridge(config);                                             // a bridge term: the flips see the shifted action
    const Type S0 = S(config);
    NSL::MCMC::MarkovState<Type> state(config, S0, 0.7, 5, true);

    std::vector<double> expected(Nx);
    for (NSL::size_t k = 0; k < Nx; ++k) {
        NSL::Configuration<Type> c{{"phi", flip.reflect(phi, k)}};
        expected[k] = std::min(1.0, std::exp(-NSL::real(S(c) - S0)));
    }
    const NSL::Tensor<Type> original(phi, true);
    const int trials = 6000;
    for (int n = 0; n < trials; ++n) {
        NSL::MCMC::MarkovState<Type> out = flip(state, S);
        // the returned state carries the action value of its configuration and the input's HMC bookkeeping
        if (n < 20) {
            REQUIRE(std::abs(NSL::real(out.actionValue - S(out.configuration))) <= 1e-10 * std::abs(NSL::real(S0)));
            REQUIRE(out.markovTime == 5);
            REQUIRE(out.accepted == true);
        }
    }
    // the input state (and the field it shares with the test) is never modified
    REQUIRE(torch::equal(torch::Tensor(state.configuration["phi"]), torch::Tensor(original)));
    REQUIRE(torch::equal(torch::Tensor(phi), torch::Tensor(original)));
    const auto & st = flip.stats();
    REQUIRE(st.proposed == NSL::size_t(trials));
    REQUIRE(st.failed == 0);
    for (NSL::size_t k = 0; k < Nx; ++k) {
        const double n = double(st.proposedPerMode[k]), f = double(st.acceptedPerMode[k]) / n;
        const double sigma = std::sqrt(std::max(expected[k] * (1 - expected[k]), 1e-4) / n);
        INFO("mode " << k << ": " << n << " proposals, acceptance " << f << ", expected " << expected[k]);
        REQUIRE(n > 0.2 * trials);                              // uniform choice of the mode
        REQUIRE(std::abs(f - expected[k]) < 5 * sigma);
    }
}

TEST_CASE("ModeFlip: off, and in RadialHMC", "[MarkovChain,modeflip]") {
    NSL::Parameter params = ringParams();
    NSL::Lattice::Ring<Type> lattice(Nx);
    const NSL::Tensor<double> V = ringV(4.);
    NSL::Action::PPPGaugeAction<Type> S_gauge(V, params);
    NSL::Action::HubbardFermionAction<Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type, decltype(lattice)>>
        S_fermion(lattice, params);
    NSL::Action::Action S = S_gauge + S_fermion;

    // 0 proposals: the state is returned and no random number is drawn
    NSL::MCMC::ModeFlip<Type> off(NSL::MCMC::interactionModes(V), 0);
    NSL::Configuration<Type> config{{"phi", gaussianField(0.5)}};
    NSL::MCMC::MarkovState<Type> state(config, S(config), 1.0, 0, true);
    torch::manual_seed(5);
    const double reference = torch::rand({1}, torch::kDouble).item<double>();
    torch::manual_seed(5);
    NSL::MCMC::MarkovState<Type> same = off(state, S);
    REQUIRE(torch::rand({1}, torch::kDouble).item<double>() == reference);
    REQUIRE(torch::equal(torch::Tensor(same.configuration["phi"]), torch::Tensor(config["phi"])));
    REQUIRE(off.stats().proposed == 0);

    // in RadialHMC: 3 proposals in every combined update step (step() is what Autotune uses)
    NSL::MCMC::ModeFlip<Type> flips(NSL::MCMC::interactionModes(V), 3);
    NSL::Integrator::ExactFourierAcceleration integrator(S, S_gauge, M_PI / 2, 4);
    const std::string file = "test_modeFlip.h5";
    std::filesystem::remove(file);
    {
        NSL::H5IO h5(file);
        NSL::MCMC::RadialHMC hmc(integrator, S, h5, NSL::MCMC::NoPreTrajectory{}, NSL::MCMC::NoPostStore{}, flips);
        for (int n = 0; n < 4; ++n) {
            state = hmc.step(state, 1, 1, 0.1);
            REQUIRE(std::abs(NSL::real(state.actionValue - S(state.configuration))) <= 1e-9 * std::abs(NSL::real(state.actionValue)));
        }
    }
    std::filesystem::remove(file);
    REQUIRE(flips.stats().proposed == 12);
}
