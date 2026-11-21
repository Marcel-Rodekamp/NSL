#ifndef NSL_BRIDGE_LINK_TPP
#define NSL_BRIDGE_LINK_TPP

/*! \file bridgeLink.tpp
 *  \brief Bridge link for charge-basis HMC: finite variance of observables quadratic in the propagator.
 *
 *  Why: the weight w(phi) = e^{-S_g} |det M_p(phi)|^2 vanishes quadratically at zeros of det M_p, where the
 *  propagator grows like 1/det. Estimators linear in M_p^{-1} keep a finite variance, but estimators quadratic
 *  in it (equal-time spin/charge correlations, every two-body correlator) have an infinite variance (Shi and
 *  Zhang, Phys. Rev. E 93, 033303 (2016)). The bridge link removes the zeros from the sampled weight and
 *  restores the target by reweighting.
 *
 *  Construction (finite temperature, charge basis; a sum of sign-free terms as in the "exact bridge link" of
 *  Wan and Zhang, Phys. Rev. E 112, 065309 (2025)): the bridge operator at time slice b is
 *      Lambda = 1/(2 Nx) sum_{i, x = +-1} e^{i theta x q_i},   q_i = n_{i up} + n_{i down} - 1,
 *  i.e. the fermion weight at the field shifted by c_{i,x} = theta x e_{b,i}. The chain samples the pair
 *  (phi, k = (i, x)) with weight e^{-S_g(phi)} |det M_p(phi + c_k)|^2:
 *   - HMC (and radial) updates of phi at fixed k: HubbardFermionAction::setShift(c_k);
 *   - a heat-bath update of k at fixed phi before every HMC trajectory (this class):
 *         P(k | phi) = R_k / sum_k' R_k',   R_k = |det M_p(phi + c_k)|^2 / |det M_p(phi)|^2
 *                                                 = |1 + (e^{i theta x} - 1) n_i(b)|^2.
 *     n_x(t) = -i d log det M_p / d phi_{t,x} is the generalized density <c^+_x c_x> of the configuration:
 *     det M_p is affine in e^{i phi_{t,x}}, so shifting phi_{t,x} by Delta multiplies det M_p by
 *     1 + (e^{i Delta} - 1) n_x(t) exactly. With Re n = 1/2 (below), n = 1/2 + i y with y = 1/2 dS_F/dphi
 *     (the fermion action -log|det M_p|^2 at the unshifted field), so the ratios follow from the force:
 *         R_{i,+-} = (cos(theta/2) -+ 2 y_i(b) sin(theta/2))^2.
 *  The marginal of phi is e^{-S_g} |det M_p(phi)|^2 Lambda(phi), Lambda(phi) = mean_k R_k(phi); the stored
 *  configurations are phi (unshifted), so measurements are taken as usual and
 *      <O> = < O / Lambda > / < 1 / Lambda >.
 *  Lambda is computed in the analysis from the measured equal-time propagator at slice b (n = 1 - G(b,b)).
 *
 *  Why the variance is finite (mu = 0, bipartite lattice): on every configuration Re n_x(t) = 1/2, so with
 *  n = 1/2 + i y,  R_{i,+-} = (cos(theta/2) -+ 2 y_i sin(theta/2))^2  and
 *      Lambda = cos^2(theta/2) + (4 sin^2(theta/2) / Nx) sum_i y_i(b)^2  >=  cos^2(theta/2).
 *  So 1/Lambda <= 1/cos^2(theta/2) is bounded, and near a zero of det M_p, where y ~ 1/det, Lambda grows like
 *  1/det^2: O/Lambda stays bounded for O up to quadratic in the propagator. theta in (0, pi); pi/2 by default.
 *
 *  Requirements: charge basis (HubbardExp), mu = 0, bipartite lattice, real field (offset 0): there
 *  |det M_p|^2 is the full fermion weight and every R_k is non-negative. Element access is on the CPU.
 */

#include "Tensor/Factory/like.tpp"
#include "complex.hpp"
#include "concepts.hpp"
#include "logger.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace NSL::MCMC {

template<NSL::Concept::isNumber Type, typename FermionActionType>
class BridgeLink {
    public:

    //! \p fermionAction: the fermion action whose copies sample the chain; this class keeps its own copy for
    //! the heat bath and installs the shared shift in \p fermionAction (do this before the action is copied
    //! into an NSL::Action::Action or an integrator). \p slice: the time slice b, \p theta: the phase.
    BridgeLink(FermionActionType & fermionAction, NSL::size_t Nt, NSL::size_t Nx, double theta, NSL::size_t slice,
               NSL::Device device = NSL::Device("CPU")) :
        theta_(theta), slice_(slice), Nx_(Nx),
        shift_(std::make_shared<NSL::Tensor<Type>>(device, Nt, Nx)),
        r_(1)
    {
        if (!(theta > 0 && theta < M_PI)) {
            throw std::invalid_argument(fmt::format("BridgeLink: theta must lie in (0, pi), got {}", theta));
        }
        if (slice >= Nt) {
            throw std::invalid_argument(fmt::format("BridgeLink: slice {} outside 0..{}", slice, Nt - 1));
        }
        (*shift_) = Type(0);
        fermionAction.setShift(shift_);
        action_ = std::make_shared<FermionActionType>(fermionAction);
        action_->setShift(nullptr);                        // the heat bath needs the unshifted field
    }

    //! bridge ratios R_k = |det M_p(phi + c_k)|^2 / |det M_p(phi)|^2, k = 2 i + (x < 0), at the configuration phi
    std::vector<double> ratios(const NSL::Tensor<Type> & phi) {
        NSL::Tensor<Type> dS = action_->grad(phi).begin()->second;     // dS_F/dphi = 2 Im n
        const double c = std::cos(theta_ / 2), s = std::sin(theta_ / 2);
        std::vector<double> R(2 * Nx_);
        for (NSL::size_t i = 0; i < Nx_; ++i) {
            const double y = 0.5 * NSL::real(dS(slice_, i));
            R[2 * i]     = (c - 2 * y * s) * (c - 2 * y * s);       // x = +1
            R[2 * i + 1] = (c + 2 * y * s) * (c + 2 * y * s);       // x = -1
        }
        return R;
    }

    //! Lambda(phi) = mean_k R_k: the reweighting factor of the configuration is 1/Lambda
    double lambda(const NSL::Tensor<Type> & phi) {
        const std::vector<double> R = ratios(phi);
        double sum = 0;
        for (double r : R) sum += r;
        return sum / R.size();
    }

    //! heat-bath update of k at fixed phi; sets the shared shift. Returns true (the action changed).
    template<typename ConfigurationType>
    bool operator()(ConfigurationType & config) {
        const std::vector<double> R = ratios(config["phi"]);
        double total = 0;
        for (double r : R) total += r;
        if (!std::isfinite(total) || total <= 0) {
            throw std::runtime_error(fmt::format("BridgeLink: invalid bridge weights (sum {})", total));
        }
        const double u = r_.rand()[0] * total;
        NSL::size_t k = 0;
        for (double acc = R[0]; acc < u && k + 1 < R.size(); acc += R[++k]) {}
        st_->k = k;
        st_->lambda = total / R.size();
        (*shift_) = Type(0);
        (*shift_)(slice_, k / 2) = Type(theta_ * ((k % 2) ? -1. : 1.));
        return true;
    }

    //! the current term k = 2 i + (x < 0) and Lambda of the configuration of the last heat bath
    NSL::size_t term() const { return st_->k; }
    double lastLambda() const { return st_->lambda; }
    double theta() const { return theta_; }
    NSL::size_t slice() const { return slice_; }

    private:
    double theta_;
    NSL::size_t slice_;
    NSL::size_t Nx_;
    std::shared_ptr<NSL::Tensor<Type>> shift_;
    std::shared_ptr<FermionActionType> action_;
    NSL::Tensor<double> r_;
    struct State { NSL::size_t k = 0; double lambda = 1; };
    std::shared_ptr<State> st_ = std::make_shared<State>();    // shared by the copies (RadialHMC keeps one)
};

} // namespace NSL::MCMC

#endif // NSL_BRIDGE_LINK_TPP
