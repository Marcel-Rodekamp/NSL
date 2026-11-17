#ifndef NSL_PPP_GAUGE_ACTION_TPP
#define NSL_PPP_GAUGE_ACTION_TPP

#include <stdexcept>
#include <string>

#include "../action.tpp"
#include "hubbard.tpp"
#include "PPP_Interaction.tpp"
#include "LinAlg.hpp"

namespace NSL::Action {

//! PPP Gauge Action
/*!
 * The PPP interaction
 *  \f[ H_V = \frac{1}{2}\sum_{xy} V_{xy}\, q_x q_y , \qquad q_x = n_{x\uparrow}+n_{x\downarrow}-1, \f]
 * (see NSL::PPP for the interaction matrix V, \f$V_{xx}=U\f$) is decoupled in
 * the charge channel on every time slice,
 *  \f[ e^{-\delta H_V} \propto \int d\phi_t\,
 *      e^{-\frac{1}{2}\phi_t^T (\delta V)^{-1}\phi_t + i\phi_t\cdot q} , \f]
 * which gives the gauge action
 *  \f[ S(\Phi) = \frac{1}{2}\sum_t \phi_t^T (\delta V)^{-1} \phi_t , \f]
 * where
 *  - \f(\delta = \frac{\beta}{N_t}\f) is the lattice spacing
 *  - \f(\beta\f) is the inverse temperature
 *  - \f(N_t\f) is the number of time slices (troterization)
 *
 * V must be symmetric positive definite. For \f$V = U\,\mathbb{1}\f$ this is
 * exactly HubbardGaugeAction.
 *
 * \warning The field couples as \f$\exp(\pm i\phi)\f$ to the charge, i.e. this
 * action has to be combined with a CHARGE basis fermion matrix
 * (NSL::FermionMatrix::HubbardExp). The SPIN basis matrix couples
 * \f$\exp(\pm\phi)\f$ to the spin; together with this action it would simulate
 * \f$-\frac{1}{2}\sum_{xy}V_{xy}s_xs_y\f$, which only agrees with the PPP model
 * for the on-site term.
 * */
template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType = Type>
class PPPGaugeAction :
    public BaseAction<Type, TensorType>
{
    public:

    //! \param V      symmetric, positive-definite Nx x Nx interaction matrix in units
    //!               of the hopping (V_xx = U), e.g. from NSL::PPP::interactionMatrixFromYAML
    //! \param params needs "beta" and "Nt" (or "delta"); "device" is used if present
	PPPGaugeAction(const NSL::Tensor<double> & V, NSL::Parameter & params, const std::string & fieldName = "phi");

    //! Reads V from the `PPP` block of an NSL yml file (see NSL::PPP)
	PPPGaugeAction(const YAML::Node & yml, NSL::Parameter & params, const std::string & fieldName = "phi") :
        PPPGaugeAction(NSL::PPP::interactionMatrixFromYAML(yml), params, fieldName)
    {}

    // We import the eval/grad/force functions from the BaseAction such
    // that we do not need to reimplement the Configuration based versions
    // We don't understand why this is not automatically done, probably due
    // to BaseAction being an abstract base class
    using BaseAction<Type,TensorType>::eval;
    using BaseAction<Type,TensorType>::grad;
    using BaseAction<Type,TensorType>::force;

    Configuration<TensorType> force(const Tensor<TensorType>& phi);
    Configuration<TensorType> grad(const Tensor<TensorType>& phi);
    Type eval(const Tensor<TensorType>& phi);

    //! Overwrites phi (shape Nt x Nx) with a sample of the gauge action alone,
    //! i.e. real \f$\phi_t \sim \mathcal{N}(0, \delta V)\f$ on every time slice.
    void sample(NSL::Tensor<TensorType> & phi) const;

    //! \f$\delta V\f$
    const NSL::Tensor<Type> & Vtilde() const { return Vtilde_; }
    //! \f$(\delta V)^{-1}\f$
    const NSL::Tensor<Type> & invVtilde() const { return invVtilde_; }
    //! symmetric square root \f$(\delta V)^{1/2}\f$
    const NSL::Tensor<Type> & sqrtVtilde() const { return sqrtVtilde_; }
    //! smallest eigenvalue of V (units of t)
    double lambdaMin() const { return lambdaMin_; }
    //! largest eigenvalue of V (units of t)
    double lambdaMax() const { return lambdaMax_; }

    protected:
    NSL::Parameter params_;
    double delta_;
    double lambdaMin_;
    double lambdaMax_;
    NSL::Tensor<Type> Vtilde_;
    NSL::Tensor<Type> invVtilde_;
    NSL::Tensor<Type> sqrtVtilde_;
};

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
PPPGaugeAction<Type, TensorType>::PPPGaugeAction(const NSL::Tensor<double> & V, NSL::Parameter & params, const std::string & fieldName) :
    BaseAction<Type, TensorType>(fieldName),
    params_(params)
{
    // delta = beta/Nt, defined exactly as in NSL::Hubbard::tilde
    if (!params.contains("delta")) {
        NSL::size_t Nt = params["Nt"];
        Type beta = params["beta"];
        params["delta"] = beta/Nt;
    }
    delta_ = NSL::real(Type(params["delta"]));

    // All linear algebra on V is done once, in double precision on the CPU
    torch::Tensor Vd = torch::Tensor(V).to(torch::kCPU).to(torch::kFloat64);
    if (Vd.dim() != 2 || Vd.size(0) != Vd.size(1)) {
        throw std::invalid_argument("NSL::Action::PPPGaugeAction: V must be a square (Nx, Nx) matrix");
    }
    if (params.contains("Nx") && NSL::size_t(params["Nx"]) != NSL::size_t(Vd.size(0))) {
        throw std::invalid_argument(fmt::format(
            "NSL::Action::PPPGaugeAction: V is {0}x{0} but Nx = {1}", Vd.size(0), NSL::size_t(params["Nx"])
        ));
    }
    const double scale = std::max(1., Vd.abs().max().item<double>());
    if ((Vd - Vd.transpose(0,1)).abs().max().item<double>() > 1e-12 * scale) {
        throw std::invalid_argument("NSL::Action::PPPGaugeAction: V must be symmetric");
    }

    auto [lambda, Q] = torch::linalg_eigh(Vd, "L");
    lambdaMin_ = lambda.min().item<double>();
    lambdaMax_ = lambda.max().item<double>();
    if (!(lambdaMin_ > 0.)) {
        throw std::invalid_argument(fmt::format(
            "NSL::Action::PPPGaugeAction: V is not positive definite (smallest eigenvalue {}); "
            "the charge-channel Hubbard-Stratonovich transformation requires V > 0", lambdaMin_
        ));
    }

    const torch::Tensor dl = delta_ * lambda;
    const torch::Tensor Qt = Q.transpose(0,1);
    torch::Tensor Vt  = delta_ * Vd;
    torch::Tensor iVt = Q.matmul(torch::diag(1./dl)).matmul(Qt);
    torch::Tensor sVt = Q.matmul(torch::diag(torch::sqrt(dl))).matmul(Qt);
    // remove round-off asymmetry
    iVt = 0.5 * (iVt + iVt.transpose(0,1));
    sVt = 0.5 * (sVt + sVt.transpose(0,1));

    const auto options = torch::TensorOptions().dtype<Type>();
    NSL::Device device("CPU");
    if (params.contains("device")) {
        device = params["device"].template to<NSL::Device>();
    }
    Vtilde_     = NSL::Tensor<Type>(Vt.to(options)).to(device);
    invVtilde_  = NSL::Tensor<Type>(iVt.to(options)).to(device);
    sqrtVtilde_ = NSL::Tensor<Type>(sVt.to(options)).to(device);
}

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
Type PPPGaugeAction<Type, TensorType>::eval(const Tensor<TensorType>& phi){
    // (phi (δV)^{-1})_{t,x} = Σ_y [(δV)^{-1}]_{xy} phi_{t,y}  as V is symmetric
    return (phi * NSL::LinAlg::mat_mul(phi, invVtilde_)).sum() / 2.;
}

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
Configuration<TensorType> PPPGaugeAction<Type, TensorType>::force(const Tensor<TensorType>& phi){
    return Configuration<TensorType>{{this->configKey_, -NSL::LinAlg::mat_mul(phi, invVtilde_)}};
}

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
Configuration<TensorType> PPPGaugeAction<Type, TensorType>::grad(const Tensor<TensorType>& phi){
    return Configuration<TensorType>{{this->configKey_, NSL::LinAlg::mat_mul(phi, invVtilde_)}};
}

template<NSL::Concept::isNumber Type, NSL::Concept::isNumber TensorType>
void PPPGaugeAction<Type, TensorType>::sample(NSL::Tensor<TensorType> & phi) const {
    torch::Tensor tphi = phi;
    torch::Tensor z = torch::randn(
        tphi.sizes(), torch::TensorOptions().dtype(torch::kFloat64).device(tphi.device())
    ).to(tphi.dtype());
    phi = NSL::Tensor<TensorType>(z.matmul(torch::Tensor(sqrtVtilde_).to(tphi.device())));
}

} // namespace NSL::Action

#endif // NSL_PPP_GAUGE_ACTION_TPP
