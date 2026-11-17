#include "../test.hpp"

#include <cmath>
#include <numbers>

// Tests of NSL::Integrator::ExactFourierAcceleration on the PPP model
// (PPP gauge action + charge-basis fermions on a 4-site ring).

namespace {

using CType = NSL::complex<double>;

NSL::Parameter makeParams(double beta, NSL::size_t Nt, NSL::size_t Nx, double U){
    NSL::Parameter params;
    params["beta"] = beta;
    params["Nt"]   = Nt;
    params["Nx"]   = Nx;
    params["U"]    = U;
    params["mu"]   = 0.;
    return params;
}

// screened Ohno interaction on a unit square (t = 2.4 eV, U = 8 eV, eps = 2, 1.4 A)
NSL::Tensor<double> squareV(){
    NSL::Tensor<double> pos(4, 2);
    pos(0,0) = 0.; pos(0,1) = 0.;
    pos(1,0) = 1.; pos(1,1) = 0.;
    pos(2,0) = 1.; pos(2,1) = 1.;
    pos(3,0) = 0.; pos(3,1) = 1.;
    NSL::PPP::Parameters p;
    p.form          = NSL::PPP::Form::Ohno;
    p.U             = 8.0/2.4;
    p.hoppingEnergy = 2.4;
    p.lengthUnit    = 1.4;
    p.dielectric    = 2.0;
    return NSL::PPP::interactionMatrix(pos, p);
}

struct PPPRing {
    using Type          = CType;
    using LatticeType   = NSL::Lattice::Ring<Type>;
    using GaugeAction   = NSL::Action::PPPGaugeAction<Type>;
    using FermionAction = NSL::Action::HubbardFermionAction<Type, LatticeType, NSL::FermionMatrix::HubbardExp<Type, LatticeType>>;

    NSL::Parameter params;
    LatticeType lattice;
    GaugeAction Sg;
    FermionAction Sf;
    NSL::Action::Action<GaugeAction, FermionAction> S;

    PPPRing(NSL::size_t Nt, double beta):
        params(makeParams(beta, Nt, 4, 8.0/2.4)),
        lattice(4),
        Sg(squareV(), params),
        Sf(lattice, params),
        S(Sg, Sf)
    {}

    NSL::Configuration<Type> sample(double offset = 0.){
        NSL::Configuration<Type> q{{"phi", NSL::Tensor<Type>(NSL::size_t(params["Nt"]), NSL::size_t(4))}};
        Sg.sample(q["phi"]);
        q["phi"].imag() = offset;
        return q;
    }
};

torch::Tensor field(const NSL::Configuration<CType> & c){ return c.at("phi"); }

double maxAbsDiff(const torch::Tensor & a, const torch::Tensor & b){
    return (a - b).abs().max().item<double>();
}

template<typename IntegratorType, typename ActionType>
double hamiltonian(IntegratorType & integrator, ActionType & S, NSL::Configuration<CType> & q, const NSL::Configuration<CType> & p){
    return NSL::real(S(q)) + NSL::real(NSL::Integrator::kineticEnergy(integrator, p));
}

} // namespace

TEST_CASE( "EFA: the gauge action alone is integrated exactly", "[Integrator,EFA]" ){
    NSL::setSeed(11);
    PPPRing m(8, 2.0);
    NSL::Action::Action<PPPRing::GaugeAction> Sg(m.Sg);

    const double T = std::numbers::pi/2;
    NSL::Integrator::ExactFourierAcceleration efa(Sg, m.Sg, T, 1);

    auto q0 = m.sample();
    auto p0 = efa.sampleMomentum(q0);
    auto [q1, p1] = efa(q0, p0);

    // exact rotation by pi/2: phi -> (δV) p, p -> -(δV)^{-1} phi
    const torch::Tensor Vt  = torch::real(torch::Tensor(m.Sg.Vtilde()));
    const torch::Tensor iVt = torch::real(torch::Tensor(m.Sg.invVtilde()));
    REQUIRE( maxAbsDiff(field(q1), torch::real(field(p0)).matmul(Vt)) < 1e-12 );
    REQUIRE( maxAbsDiff(field(p1), -torch::real(field(q0)).matmul(iVt)) < 1e-12 );

    const double H0 = hamiltonian(efa, Sg, q0, p0);
    const double H1 = hamiltonian(efa, Sg, q1, p1);
    REQUIRE( std::abs(H1 - H0) < 1e-12 * std::abs(H0) );

    // the same single step with the unit-mass leapfrog is far off
    NSL::Integrator::LeapfrogRealForce leapfrog(Sg, T, 1);
    auto pl = NSL::Integrator::sampleMomentum(leapfrog, q0);
    auto [ql1, pl1] = leapfrog(q0, pl);
    REQUIRE( std::abs(hamiltonian(leapfrog, Sg, ql1, pl1) - hamiltonian(leapfrog, Sg, q0, pl)) > 1e-3 );
}

TEST_CASE( "EFA: trajectories are reversible and keep the imaginary offset", "[Integrator,EFA]" ){
    NSL::setSeed(12);
    PPPRing m(8, 2.0);
    const double T = 1.3;
    const NSL::size_t Nmd = 7;
    NSL::Integrator::ExactFourierAcceleration efa(m.S, m.Sg, T, Nmd);
    NSL::Integrator::ExactFourierAcceleration efaBack(m.S, m.Sg, T, Nmd, /*backward*/true);

    for (double offset : {0., 0.3}) {
        INFO( "offset " << offset );
        auto q0 = m.sample(offset);
        auto p0 = efa.sampleMomentum(q0);
        REQUIRE( torch::imag(field(p0)).abs().max().item<double>() == 0. );

        auto [q1, p1] = efa(q0, p0);
        REQUIRE( maxAbsDiff(torch::imag(field(q1)), torch::imag(field(q0))) == 0. );
        REQUIRE( torch::imag(field(p1)).abs().max().item<double>() == 0. );
        REQUIRE( maxAbsDiff(field(q1), field(q0)) > 0.1 );

        // momentum flip
        NSL::Configuration<CType> mp1{{"phi", NSL::Tensor<CType>(-torch::Tensor(field(p1)))}};
        auto [q2, p2] = efa(q1, mp1);
        REQUIRE( maxAbsDiff(field(q2), field(q0)) < 1e-10 );
        REQUIRE( maxAbsDiff(field(p2), -field(p0)) < 1e-10 );

        // backward integration
        auto [q3, p3] = efaBack(q1, p1);
        REQUIRE( maxAbsDiff(field(q3), field(q0)) < 1e-10 );
        REQUIRE( maxAbsDiff(field(p3), field(p0)) < 1e-10 );
    }
}

TEST_CASE( "EFA: the energy violation is of second order in the step size", "[Integrator,EFA]" ){
    NSL::setSeed(13);
    PPPRing m(8, 2.0);
    const double T = std::numbers::pi/2;

    NSL::Integrator::ExactFourierAcceleration efa0(m.S, m.Sg, T, 1);
    auto q0 = m.sample();
    auto p0 = efa0.sampleMomentum(q0);
    const double H0 = hamiltonian(efa0, m.S, q0, p0);

    std::vector<double> dH;
    // asymptotic regime (coarser steps are not yet in it for this configuration)
    for (NSL::size_t Nmd : {40, 80, 160}) {
        NSL::Integrator::ExactFourierAcceleration efa(m.S, m.Sg, T, Nmd);
        auto [q1, p1] = efa(q0, p0);
        dH.push_back(std::abs(hamiltonian(efa, m.S, q1, p1) - H0));
        INFO( "Nmd = " << Nmd << ": |dH| = " << dH.back() );
        CHECK( dH.back() > 0. );
    }
    INFO( "ratios " << dH[0]/dH[1] << " " << dH[1]/dH[2] );
    REQUIRE( dH[0]/dH[1] == Approx(4.).epsilon(0.25) );
    REQUIRE( dH[1]/dH[2] == Approx(4.).epsilon(0.25) );
}

TEST_CASE( "EFA: phase-space volume is preserved", "[Integrator,EFA]" ){
    NSL::setSeed(14);
    PPPRing m(2, 1.0);
    NSL::Integrator::ExactFourierAcceleration efa(m.S, m.Sg, 1.1, 3);

    auto q0 = m.sample();
    auto p0 = efa.sampleMomentum(q0);
    const NSL::size_t n = torch::Tensor(field(q0)).numel();

    // phase-space point (Re phi, p) -> (Re phi', p')
    auto flow = [&](const torch::Tensor & z){
        NSL::Configuration<CType> q{{"phi", NSL::Tensor<CType>(torch::complex(z.slice(0,0,n).reshape({2,4}), torch::zeros({2,4}, torch::kFloat64)))}};
        NSL::Configuration<CType> p{{"phi", NSL::Tensor<CType>(torch::complex(z.slice(0,n,2*n).reshape({2,4}), torch::zeros({2,4}, torch::kFloat64)))}};
        auto [q1, p1] = efa(q, p);
        return torch::cat({torch::real(field(q1)).flatten(), torch::real(field(p1)).flatten()});
    };

    const torch::Tensor z0 = torch::cat({torch::real(field(q0)).flatten(), torch::real(field(p0)).flatten()});
    const double h = 1e-5;
    torch::Tensor J = torch::zeros({long(2*n), long(2*n)}, torch::kFloat64);
    for (long k = 0; k < long(2*n); ++k){
        torch::Tensor zp = z0.clone(), zm = z0.clone();
        zp[k] += h; zm[k] -= h;
        J.select(1, k).copy_((flow(zp) - flow(zm)) / (2*h));
    }
    const double det = torch::linalg_det(J).item<double>();
    INFO( "det J = " << det );
    REQUIRE( det == Approx(1.).epsilon(1e-6) );
}

TEST_CASE( "EFA: momenta follow exp(-kinetic energy)", "[Integrator,EFA]" ){
    NSL::setSeed(15);
    // many time slices give many independent samples of p_t ~ N(0, A)
    const NSL::size_t Nt = 20000;
    PPPRing m(Nt, 2.0);
    NSL::Integrator::ExactFourierAcceleration efa(m.S, m.Sg, 1., 1);

    auto q = m.sample();
    auto p = efa.sampleMomentum(q);
    const torch::Tensor P = torch::real(field(p));
    const torch::Tensor A = torch::real(torch::Tensor(m.Sg.invVtilde()));

    const torch::Tensor cov = P.transpose(0,1).matmul(P) / double(Nt);
    const double tol = 6. * std::sqrt(2./Nt) * A.abs().max().item<double>();
    REQUIRE( maxAbsDiff(cov, A) < tol );

    // <K> = (number of degrees of freedom)/2
    const double K = NSL::real(efa.kineticEnergy(p));
    const double dof = 4. * Nt;
    REQUIRE( std::abs(K/dof - 0.5) < 6. * std::sqrt(0.5/dof) );
}

TEST_CASE( "EFA: HMC hooks select the kinetic term of the integrator", "[Integrator,EFA]" ){
    NSL::setSeed(16);
    PPPRing m(4, 1.0);
    NSL::Integrator::ExactFourierAcceleration efa(m.S, m.Sg, 1., 2);
    NSL::Integrator::LeapfrogRealForce leapfrog(m.S, 1., 2);

    static_assert( NSL::Integrator::hasMassMatrix<decltype(efa), CType> );
    static_assert( !NSL::Integrator::hasMassMatrix<decltype(leapfrog), CType> );

    auto q = m.sample();
    auto p = NSL::Integrator::sampleMomentum(efa, q);
    const torch::Tensor P = torch::real(field(p));
    const torch::Tensor B = torch::real(torch::Tensor(m.Sg.Vtilde()));

    const double Kefa = NSL::real(NSL::Integrator::kineticEnergy(efa, p));
    REQUIRE( Kefa == Approx(0.5 * (P * P.matmul(B)).sum().item<double>()).epsilon(1e-13) );

    const double Kunit = NSL::real(NSL::Integrator::kineticEnergy(leapfrog, p));
    REQUIRE( Kunit == Approx(0.5 * (P * P).sum().item<double>()).epsilon(1e-13) );

    auto pu = NSL::Integrator::sampleMomentum(leapfrog, q);
    REQUIRE( torch::imag(field(pu)).abs().max().item<double>() == 0. );
}

TEST_CASE( "Unit-mass momenta have unit variance for real and complex fields", "[Integrator]" ){
    NSL::setSeed(17);
    struct UnitMass {};   // any integrator without a mass matrix
    UnitMass integrator;
    const long n = 200000;

    NSL::Configuration<double> qr{{"phi", NSL::Tensor<double>(n)}};
    torch::Tensor pr = NSL::Integrator::sampleMomentum(integrator, qr).at("phi");
    REQUIRE( pr.var().item<double>() == Approx(1.).epsilon(0.02) );

    NSL::Configuration<CType> qc{{"phi", NSL::Tensor<CType>(n)}};
    torch::Tensor pc = NSL::Integrator::sampleMomentum(integrator, qc).at("phi");
    REQUIRE( torch::real(pc).var().item<double>() == Approx(1.).epsilon(0.02) );
    REQUIRE( torch::imag(pc).abs().max().item<double>() == 0. );
}

TEST_CASE( "EFA: invalid mass matrices are rejected", "[Integrator,EFA]" ){
    PPPRing m(4, 1.0);
    using EFA = NSL::Integrator::ExactFourierAcceleration<PPPRing::GaugeAction, PPPRing::FermionAction>;

    torch::Tensor nonSymmetric = torch::eye(4, torch::kFloat64);
    nonSymmetric[0][1] = 0.5;
    REQUIRE_THROWS_AS( EFA(m.S, nonSymmetric, 1., 2), std::invalid_argument );

    torch::Tensor indefinite = torch::eye(4, torch::kFloat64);
    indefinite[2][2] = -1.;
    REQUIRE_THROWS_AS( EFA(m.S, indefinite, 1., 2), std::invalid_argument );

    REQUIRE_THROWS_AS( EFA(m.S, torch::eye(4, torch::kFloat64), 1., 0), std::invalid_argument );

    // wrong number of sites
    EFA efa5(m.S, torch::eye(5, torch::kFloat64), 1., 2);
    auto q = m.sample();
    REQUIRE_THROWS_AS( efa5.sampleMomentum(q), std::invalid_argument );
}
