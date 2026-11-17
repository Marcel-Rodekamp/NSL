#include "../test.hpp"

#include <cmath>
#include <limits>

// =============================================================================
// Helpers
// =============================================================================

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

double maxAbsDiff(const NSL::Tensor<CType> & a, const NSL::Tensor<CType> & b){
    return torch::abs(torch::Tensor(a) - torch::Tensor(b)).max().item<double>();
}

double maxAbs(const NSL::Tensor<CType> & a){
    return torch::abs(torch::Tensor(a)).max().item<double>();
}

// 4 sites on a unit square (ring geometry)
NSL::Tensor<double> squarePositions(){
    NSL::Tensor<double> pos(4, 2);
    pos(0,0) = 0.; pos(0,1) = 0.;
    pos(1,0) = 1.; pos(1,1) = 0.;
    pos(2,0) = 1.; pos(2,1) = 1.;
    pos(3,0) = 0.; pos(3,1) = 1.;
    return pos;
}

NSL::PPP::Parameters benchmarkParameters(NSL::PPP::Form form){
    NSL::PPP::Parameters p;
    p.form          = form;
    p.U             = 8.0/2.4;
    p.hoppingEnergy = 2.4;
    p.lengthUnit    = 1.4;
    p.dielectric    = 2.0;
    return p;
}

} // namespace

// =============================================================================
// Interaction matrices
// =============================================================================

TEST_CASE( "PPP: interaction form names", "[PPP,interaction]" ){
    using NSL::PPP::Form;
    REQUIRE( NSL::PPP::formFromString("ohno")             == Form::Ohno );
    REQUIRE( NSL::PPP::formFromString("Ohno")             == Form::Ohno );
    REQUIRE( NSL::PPP::formFromString("Mataga-Nishimoto") == Form::MatagaNishimoto );
    REQUIRE( NSL::PPP::formFromString("mataga_nishimoto") == Form::MatagaNishimoto );
    REQUIRE( NSL::PPP::formFromString("MN")               == Form::MatagaNishimoto );
    REQUIRE( NSL::PPP::formFromString("onsite")           == Form::OnSite );
    REQUIRE( NSL::PPP::formFromString("hubbard")          == Form::OnSite );
    REQUIRE( NSL::PPP::formFromString("explicit")         == Form::Explicit );
    REQUIRE_THROWS_AS( NSL::PPP::formFromString("coulomb"), std::invalid_argument );
    for (auto f: {Form::Ohno, Form::MatagaNishimoto, Form::OnSite, Form::Explicit}){
        REQUIRE( NSL::PPP::formFromString(NSL::PPP::to_string(f)) == f );
    }
}

TEST_CASE( "PPP: interaction matrix elements", "[PPP,interaction]" ){
    using NSL::PPP::Form;
    NSL::Tensor<double> pos = squarePositions();

    for (auto form: {Form::Ohno, Form::MatagaNishimoto, Form::OnSite}){
        NSL::PPP::Parameters p = benchmarkParameters(form);
        NSL::Tensor<double> V = NSL::PPP::interactionMatrix(pos, p);
        INFO( "form = " + NSL::PPP::to_string(form) );

        for (NSL::size_t x = 0; x < 4; ++x){
            REQUIRE( double(V(x,x)) == Approx(p.U) );
            for (NSL::size_t y = 0; y < 4; ++y){
                REQUIRE( double(V(x,y)) == double(V(y,x)) );
            }
        }

        // nearest neighbours (r = 1) and the diagonal of the square (r = sqrt 2)
        for (double r: {1.0, std::sqrt(2.)}){
            const double xi = p.U * p.hoppingEnergy * p.dielectric * p.lengthUnit * r / NSL::PPP::e2_eV_Angstrom;
            double expected = 0.;
            if (form == Form::Ohno)            { expected = p.U / std::sqrt(1. + xi*xi); }
            if (form == Form::MatagaNishimoto) { expected = p.U / (1. + xi); }
            const NSL::size_t y = (r == 1.0) ? 1 : 2;
            REQUIRE( double(V(0,y)) == Approx(expected).epsilon(1e-14) );
        }
    }

    // Both forms approach the screened Coulomb tail e^2/(eps r) at large r (in units of t)
    for (auto form: {Form::Ohno, Form::MatagaNishimoto}){
        NSL::PPP::Parameters p = benchmarkParameters(form);
        const double r = 1e4;
        const double coulomb = NSL::PPP::e2_eV_Angstrom / (p.dielectric * r * p.lengthUnit) / p.hoppingEnergy;
        REQUIRE( NSL::PPP::offDiagonal(p, r) == Approx(coulomb).epsilon(1e-3) );
    }

    // coincident sites are rejected
    NSL::Tensor<double> bad(2, 2);
    bad = 0.;
    REQUIRE_THROWS_AS( NSL::PPP::interactionMatrix(bad, benchmarkParameters(Form::Ohno)), std::invalid_argument );
}

TEST_CASE( "PPP: interaction matrix from yml", "[PPP,interaction]" ){
    const std::string base = R"(
system:
  nions: 4
  U: 3.3333333333333335
  positions:
  - [0, 0, 0]
  - [1, 0, 0]
  - [1, 1, 0]
  - [0, 1, 0]
)";

    SECTION("ohno"){
        YAML::Node yml = YAML::Load(base + R"(
PPP:
  form: ohno
  hopping energy: 2.4
  length unit: 1.4
  dielectric: 2.0
)");
        NSL::Tensor<double> V    = NSL::PPP::interactionMatrixFromYAML(yml);
        NSL::Tensor<double> Vref = NSL::PPP::interactionMatrix(squarePositions(), benchmarkParameters(NSL::PPP::Form::Ohno));
        REQUIRE( torch::allclose(torch::Tensor(V), torch::Tensor(Vref), 1e-14, 1e-14) );
    }

    SECTION("explicit"){
        YAML::Node yml = YAML::Load(base + R"(
PPP:
  form: explicit
  V:
  - [3.3333333333333335, 1.0, 0.5, 1.0]
  - [1.0, 3.3333333333333335, 1.0, 0.5]
  - [0.5, 1.0, 3.3333333333333335, 1.0]
  - [1.0, 0.5, 1.0, 3.3333333333333335]
)");
        NSL::Tensor<double> V = NSL::PPP::interactionMatrixFromYAML(yml);
        REQUIRE( double(V(0,2)) == 0.5 );
        REQUIRE( double(V(3,0)) == 1.0 );
        REQUIRE( double(V(1,1)) == Approx(10./3.) );
    }

    SECTION("missing form"){
        YAML::Node yml = YAML::Load(base);
        REQUIRE_THROWS_AS( NSL::PPP::interactionMatrixFromYAML(yml), std::invalid_argument );
    }
}

// =============================================================================
// Gauge action
// =============================================================================

TEST_CASE( "PPP: on-site interaction reproduces the Hubbard gauge action", "[PPP,action]" ){
    NSL::setSeed(1234);
    const NSL::size_t Nt = 6, Nx = 4;
    const double U = 3.7; // deliberately not an integer
    NSL::Parameter params = makeParams(2.5, Nt, Nx, U);

    NSL::PPP::Parameters p = benchmarkParameters(NSL::PPP::Form::OnSite);
    p.U = U;
    NSL::Action::PPPGaugeAction<CType>     Sppp(NSL::PPP::interactionMatrix(squarePositions(), p), params);
    NSL::Action::HubbardGaugeAction<CType> Shub(params);

    NSL::Tensor<CType> phi(Nt, Nx);
    phi.randn();   // complex fields are fine for the Gaussian part

    REQUIRE( NSL::LinAlg::abs(Sppp.eval(phi) - Shub.eval(phi)) <= 1e-12 * NSL::LinAlg::abs(Shub.eval(phi)) );
    REQUIRE( maxAbsDiff(Sppp.force(phi)["phi"], Shub.force(phi)["phi"]) <= 1e-12 * maxAbs(Shub.force(phi)["phi"]) );
    REQUIRE( maxAbsDiff(Sppp.grad(phi)["phi"],  Shub.grad(phi)["phi"])  <= 1e-12 * maxAbs(Shub.grad(phi)["phi"]) );
}

TEST_CASE( "PPP: gauge action of two sites with non-integer U", "[PPP,action]" ){
    NSL::setSeed(4321);
    const NSL::size_t Nt = 4, Nx = 2;
    const double beta = 2.0, U = 3.5;
    NSL::Parameter params = makeParams(beta, Nt, Nx, U);

    NSL::PPP::Parameters p = benchmarkParameters(NSL::PPP::Form::MatagaNishimoto);
    p.U = U;
    NSL::Tensor<double> pos(2, 2);
    pos = 0.;
    pos(1,0) = 1.;
    NSL::Tensor<double> V = NSL::PPP::interactionMatrix(pos, p);

    NSL::Action::PPPGaugeAction<CType> S(V, params);

    NSL::Tensor<CType> phi(Nt, Nx);
    phi.randn();

    // S = 1/2 Σ_t φ_t^T (δV)^{-1} φ_t with the analytic inverse of [[d,o],[o,d]]
    const double delta = beta/Nt;
    const double d = delta * double(V(0,0)), o = delta * double(V(0,1));
    const double det = d*d - o*o, id = d/det, io = -o/det;
    CType expected = 0.;
    for (NSL::size_t t = 0; t < Nt; ++t){
        CType x0 = phi(t,0), x1 = phi(t,1);
        expected += x0*(id*x0 + io*x1) + x1*(io*x0 + id*x1);
    }
    expected /= 2.;

    REQUIRE( NSL::LinAlg::abs(S.eval(phi) - expected) <= 1e-12 * NSL::LinAlg::abs(expected) );
    REQUIRE( S.lambdaMin() == Approx(double(V(0,0)) - double(V(0,1))) );
    REQUIRE( S.lambdaMax() == Approx(double(V(0,0)) + double(V(0,1))) );
}

TEST_CASE( "PPP: gauge force agrees with finite differences and is not aliased", "[PPP,action,force]" ){
    NSL::setSeed(99);
    const NSL::size_t Nt = 5, Nx = 4;
    NSL::Parameter params = makeParams(3.0, Nt, Nx, 8.0/2.4);
    NSL::Action::PPPGaugeAction<CType> S(
        NSL::PPP::interactionMatrix(squarePositions(), benchmarkParameters(NSL::PPP::Form::Ohno)), params
    );

    NSL::Tensor<CType> phi(Nt, Nx);
    phi.randn();
    phi.imag() = 0;

    NSL::Tensor<CType> F = S.force(phi)["phi"];
    NSL::Tensor<CType> G = S.grad(phi)["phi"];
    REQUIRE( maxAbsDiff(F, -G) == 0. );

    // central differences; S is quadratic, so they are exact up to round-off
    const double h = 1e-3;
    for (NSL::size_t t = 0; t < Nt; ++t){
        for (NSL::size_t x = 0; x < Nx; ++x){
            NSL::Tensor<CType> pp(phi, true), pm(phi, true);
            pp(t,x) += h;
            pm(t,x) -= h;
            const double fd = NSL::real(S.eval(pp) - S.eval(pm)) / (2*h);
            REQUIRE( fd == Approx(NSL::real(CType(G(t,x)))).epsilon(1e-8).margin(1e-9) );
        }
    }

    // the returned gradient must not change when the action is used again
    const CType g00 = G(0,0);
    NSL::Tensor<CType> other(Nt, Nx);
    other.randn();
    S.eval(other);
    S.grad(other);
    S.force(other);
    REQUIRE( CType(G(0,0)) == g00 );
}

TEST_CASE( "PPP: invalid interaction matrices are rejected", "[PPP,action]" ){
    NSL::Parameter params = makeParams(1.0, 4, 2, 1.0);

    NSL::Tensor<double> notPD(2, 2);
    notPD(0,0) = 1.; notPD(1,1) = 1.;
    notPD(0,1) = 2.; notPD(1,0) = 2.;   // eigenvalues 3 and -1
    REQUIRE_THROWS_AS( NSL::Action::PPPGaugeAction<CType>(notPD, params), std::invalid_argument );

    NSL::Tensor<double> notSym(2, 2);
    notSym(0,0) = 2.; notSym(1,1) = 2.;
    notSym(0,1) = 1.; notSym(1,0) = 0.5;
    REQUIRE_THROWS_AS( NSL::Action::PPPGaugeAction<CType>(notSym, params), std::invalid_argument );

    NSL::Tensor<double> wrongSize(3, 3);
    wrongSize = 0.;
    wrongSize(0,0) = 1.; wrongSize(1,1) = 1.; wrongSize(2,2) = 1.;
    REQUIRE_THROWS_AS( NSL::Action::PPPGaugeAction<CType>(wrongSize, params), std::invalid_argument );
}

TEST_CASE( "PPP: sample() draws phi_t ~ N(0, delta V)", "[PPP,action]" ){
    NSL::setSeed(2024);
    const NSL::size_t Nt = 40000, Nx = 4;
    const double beta = 4.0;
    NSL::Parameter params = makeParams(beta, Nt, Nx, 8.0/2.4);
    NSL::Tensor<double> V = NSL::PPP::interactionMatrix(squarePositions(), benchmarkParameters(NSL::PPP::Form::Ohno));
    NSL::Action::PPPGaugeAction<CType> S(V, params);

    NSL::Tensor<CType> phi(Nt, Nx);
    S.sample(phi);

    torch::Tensor tphi = torch::Tensor(phi);
    REQUIRE( torch::abs(torch::imag(tphi)).max().item<double>() == 0. );

    // empirical covariance over time slices vs delta V
    torch::Tensor re  = torch::real(tphi);
    torch::Tensor cov = re.transpose(0,1).matmul(re) / double(Nt);
    torch::Tensor ref = torch::real(torch::Tensor(S.Vtilde()));
    // statistical error of a covariance estimate ~ sqrt(2/Nt) * scale
    const double tol = 6. * std::sqrt(2./Nt) * ref.abs().max().item<double>();
    REQUIRE( (cov - ref).abs().max().item<double>() < tol );
}

TEST_CASE( "PPP: full action force (PPP gauge + charge-basis fermions) vs finite differences", "[PPP,action,force]" ){
    NSL::setSeed(7);
    using Type = CType;
    const NSL::size_t Nt = 8, Nx = 4;
    NSL::Parameter params = makeParams(2.0, Nt, Nx, 8.0/2.4);

    NSL::Lattice::Ring<Type> lattice(Nx);
    NSL::Action::PPPGaugeAction<Type> Sg(
        NSL::PPP::interactionMatrix(squarePositions(), benchmarkParameters(NSL::PPP::Form::Ohno)), params
    );
    NSL::Action::HubbardFermionAction<Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>> Sf(lattice, params);
    NSL::Action::Action S = Sg + Sf;

    NSL::Configuration<Type> config{{"phi", NSL::Tensor<Type>(Nt, Nx)}};
    Sg.sample(config["phi"]);

    NSL::Configuration<Type> gradS = S.grad(config);
    REQUIRE( torch::abs(torch::imag(torch::Tensor(S.force(config)["phi"]))).max().item<double>() == 0. );

    const double h = 1e-5;
    for (NSL::size_t t = 0; t < Nt; ++t){
        for (NSL::size_t x = 0; x < Nx; ++x){
            NSL::Configuration<Type> cp(config, true), cm(config, true);
            cp["phi"](t,x) += h;
            cm["phi"](t,x) -= h;
            const double fd = NSL::real(S(cp) - S(cm)) / (2*h);
            const double an = NSL::real(Type(gradS["phi"](t,x)));
            INFO( fmt::format("t={} x={} FD={} grad={}", t, x, fd, an) );
            REQUIRE( fd == Approx(an).epsilon(1e-5).margin(1e-7) );
        }
    }
}
