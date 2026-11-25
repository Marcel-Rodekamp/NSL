#include "../test.hpp"
#include <highfive/H5File.hpp>
#include <iostream>

template<typename Type,template<typename Type_, typename Lattice_> class HFM >
void test_force();

// =============================================================================
// Test Cases
// =============================================================================

COMPLEX_NSL_TEST_CASE( "Action: tests force routine and compares to finite differencing", "[Action,force,default]" ) {
    INFO("Hubbard Exponential");
    test_force<TestType,NSL::FermionMatrix::HubbardExp>();
    // HubbardDiag is outdated and no longer provides the interface HubbardFermionAction
    // expects (bipartite_, public mu_), so it is not tested here.
}

//=======================================================================
// Implementation Details
//=======================================================================

//! the parameters of the test system: Hubbard model on a ring, beta = 2, U = 3, mu = 0, Nt time slices
template<typename Type>
NSL::Parameter parameters(NSL::size_t Nt){
    NSL::Parameter params;
    params["beta"] = Type(2.);
    params["U"] = Type(3.);
    params["mu"] = 0.;
    params["Nt"] = Nt;
    return params;
}

//! the action S = S_gauge + S_fermion of the test system at phi (Nt x Nx)
template<typename Type,template<typename Type_, typename Lattice_> class HFM >
Type action(const NSL::Tensor<Type> & phi){
    NSL::Lattice::Ring<Type> lattice(phi.shape(1));
    NSL::Parameter params = parameters<Type>(phi.shape(0));
    NSL::Action::HubbardGaugeAction<Type> S_gauge(params);
    NSL::Action::HubbardFermionAction<Type,decltype(lattice),HFM<Type,decltype(lattice)>> S_fermion(lattice, params);
    NSL::Action::Action S = S_gauge + S_fermion;
    NSL::Configuration<Type> config{{"phi", phi}};
    return S(config);
}

template<typename Type,template<typename Type_, typename Lattice_> class HFM >
void test_force(){
    //    Number of time slices
    NSL::size_t Nt = 8;
    //    Number of ions (spatial sites)
    NSL::size_t Nx = 2;

    // Define the lattice geometry of interest
    NSL::Lattice::Ring<Type> lattice(Nx);
    NSL::Parameter params = parameters<Type>(Nt);

    NSL::Logger::info("Setting up a Hubbard action with beta=2, Nt={}, U=3, on a ring with {} sites.", Nt, Nx);

    NSL::Action::HubbardGaugeAction<Type> S_gauge(params);
    NSL::Action::HubbardFermionAction<Type,decltype(lattice),HFM<Type,decltype(lattice)>> S_fermion(lattice, params);

    // Initialize the action
    NSL::Action::Action S = S_gauge + S_fermion;

    // Initialize a configuration as starting point for the MC change
    NSL::Configuration<Type> config{
        {"phi", NSL::Tensor<Type>(Nt,Nx)}
    };
    config["phi"].randn();
    config["phi"].imag() = 0; // use purely real fields

    config["phi"] *= NSL::Hubbard::tilde<Type>(params, "U");

    NSL::Configuration<Type> gradS{
        {"phi", NSL::Tensor<Type>(Nt,Nx)}
    };
    // This is how one computes the grad of the action
    gradS = S.grad(config);

    REQUIRE( (config["phi"].imag() == S.force(config)["phi"].imag() ).all() );  // the force should all real (when chemical potential is zero)

    // Reference: central differences of the action evaluated in double precision at the same field,
    //     (S(phi + eps e_ti) - S(phi - eps e_ti)) / (2 eps),
    // with error eps^2 |d^3S| / 6 + u |S| / eps (u = 1.1e-16), whatever the precision of Type. (A finite difference
    // in Type itself cannot test complex<float>: its rounding error u |S| / eps is ~1e-2 for eps = 1e-4.)
    // The step shrinks with the slope: near a zero of det M at distance d, S_f ~ -2 log d, so d^3S ~ (dS)^3 / 2 and
    // eps = 1e-4 / max(1, |dS|) keeps the truncation error eps^2 |dS|^3 / 12 at ~1e-9 |dS| (a random field comes
    // close to a zero in roughly one of ten draws; with a fixed eps = 1e-4 the reference was then off by 0.4 at
    // dS = 800).
    // The gradient is computed in Type, so the tolerance is set by the precision of Type; it is at least 1e-7, well
    // above the error of the reference.
    typedef NSL::complex<double> Ref;
    const NSL::Tensor<Ref> phiRef = static_cast<NSL::Tensor<Ref>>(config["phi"]);
    const double tol = std::max(1e-7, 1e3 * double(std::numeric_limits<NSL::RealTypeOf<Type>>::epsilon()));
    for (int t=0; t< Nt; t++){
        for (int i=0;i<Nx;i++) {
            const double epsilon = 1e-4 / std::max(1.0, double(std::abs(gradS["phi"](t,i))));
            NSL::Tensor<Ref> plus(phiRef, true), minus(phiRef, true);
            plus(t,i) += epsilon;
            minus(t,i) -= epsilon;
            const Ref fin_diff = (action<Ref,HFM>(plus) - action<Ref,HFM>(minus)) / (2 * epsilon);
            const Ref grad(NSL::real(gradS["phi"](t,i)), NSL::imag(gradS["phi"](t,i)));
            const double err = std::abs(fin_diff - grad);

            INFO( fmt::format("t={},x={}",t,i) );
            INFO( fmt::format("ΔS/2ε (double) = {}",NSL::to_string(fin_diff)) );
            INFO( fmt::format("∂S             = {}",NSL::to_string(gradS["phi"](t,i))) );
            INFO( fmt::format("err = {:.3e}, tolerance {:.1e} * max(1, |ΔS/2ε|), ε = {:.1e}",err,tol,epsilon) );
            REQUIRE( err <= tol * std::max(1.0, std::abs(fin_diff)) );
        }
    }
}
