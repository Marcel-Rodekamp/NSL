#ifndef NSL_MODE_FLIP_TPP
#define NSL_MODE_FLIP_TPP

/*! \file modeFlip.tpp
 *  \brief Mode flips: Metropolis updates that reverse one mode of the field on every time slice.
 *
 *  Why: in the charge basis at mu = 0 on a bipartite lattice, R = e^{-i sum phi / 2} det M_p is real, and its sign
 *  splits the field space into sectors, separated by the zeros of det M_p. HMC crosses between them rarely (perylene,
 *  beta = 24, Nt = 288: one sector change per 30-300 trajectories, with the single- or the multi-slice bridge
 *  link), so the sector populations converge slowly. A mode flip is a large move that changes the sector often and
 *  costs one action evaluation.
 *
 *  Definition: for orthonormal modes v_k (the columns of `modes`, Nx x Nx), the flip of mode k reflects the field on
 *  every time slice t,
 *      phi_t -> phi_t - 2 (v_k . phi_t) v_k,
 *  i.e. it reverses the amplitude of mode k at all times. The reflection is an involution with unit Jacobian, so with
 *  k drawn uniformly the proposal is symmetric, and the Metropolis acceptance min(1, e^{-(S(phi') - S(phi))}) with
 *  the full action makes the update exact, for any action and any orthonormal modes.
 *  With the eigenvectors of the interaction V as modes, the gauge action S_g = 1/2 sum_t phi_t^T (delta V)^-1 phi_t is
 *  unchanged by every flip, and only the fermion action decides. For Hubbard (V = U 1) every orthonormal basis
 *  qualifies; the site basis gives phi_{t,x} -> -phi_{t,x} for all t, the global moves of Scalettar, Noack and Singh,
 *  Phys. Rev. B 44, 10502 (1991). For PPP the site flips are almost never accepted (they change S_g by O(100) on
 *  perylene), while the V-eigenmode flips are accepted 20-35 % of the time.
 *
 *  Use: an extra update of NSL::MCMC::RadialHMC, run after the radial updates of every combined update step, before
 *  the HMC trajectory; `proposals` flips per step, each of a mode drawn uniformly. It is called with the action of the
 *  HMC, so with NSL::MCMC::BridgeLink it accepts on the shifted action at the current bridge term, which is the
 *  correct update of the joint chain (phi, k). Only the real part of the field is reflected (a constant imaginary
 *  offset stays), as in the radial update. With proposals = 0 it returns at once and draws no random numbers, so a
 *  chain is the same as without the update. The state's acceptance probability and flag stay those of the last HMC or
 *  radial update (the HMC diagnostics use them); the counts of the flips are kept in stats(). The input state is
 *  never modified: an accepted flip gives the returned state new tensors.
 */

#include "Configuration/Configuration.tpp"
#include "MarkovChain/markovState.tpp"
#include "complex.hpp"
#include "concepts.hpp"
#include "logger.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace NSL::MCMC {

template<NSL::Concept::isNumber Type>
class ModeFlip {
    public:

    //! counts of the proposals, shared by the copies of the update (RadialHMC keeps a copy)
    struct Stats {
        NSL::size_t proposed = 0, accepted = 0, failed = 0;
        std::vector<NSL::size_t> proposedPerMode, acceptedPerMode;
        double acceptance() const { return proposed ? double(accepted) / double(proposed) : NAN; }
    };

    //! \p modes: Nx x Nx, column k is the mode v_k (orthonormal columns); \p proposals: flips per call (0: off);
    //! \p key: the field of the configuration that is flipped
    ModeFlip(const NSL::Tensor<double> & modes, NSL::size_t proposals, std::string key = "phi") :
        modes_(torch::Tensor(modes).to(torch::kCPU, torch::kDouble).contiguous()), proposals_(proposals),
        key_(std::move(key)), r_(2), stats_(std::make_shared<Stats>())
    {
        if (modes_.dim() != 2 || modes_.size(0) != modes_.size(1) || modes_.size(0) == 0) {
            throw std::invalid_argument("ModeFlip: the modes must be a non-empty square matrix (Nx x Nx)");
        }
        const torch::Tensor gram = torch::matmul(modes_.transpose(0, 1), modes_);
        const double dev = (gram - torch::eye(modes_.size(0), torch::kDouble)).abs().max().template item<double>();
        if (!(dev < 1e-8)) {
            throw std::invalid_argument(fmt::format("ModeFlip: the modes are not orthonormal (max |V^T V - 1| = {})", dev));
        }
        stats_->proposedPerMode.assign(modes_.size(0), 0);
        stats_->acceptedPerMode.assign(modes_.size(0), 0);
    }

    //! the field with its real part reflected along mode k on every time slice (phi: Nt x Nx)
    NSL::Tensor<Type> reflect(const NSL::Tensor<Type> & phi, NSL::size_t k) const {
        const torch::Tensor p = torch::Tensor(phi);
        const torch::Tensor re = p.is_complex() ? torch::real(p) : p;
        const torch::Tensor v = modes_.select(1, int64_t(k)).to(re.device(), re.scalar_type());
        const torch::Tensor reflected = re - 2 * torch::outer(torch::matmul(re, v), v);
        return NSL::Tensor<Type>(p.is_complex() ? torch::complex(reflected, torch::imag(p)) : reflected);
    }

    //! `proposals` Metropolis flips of the state with the action \p action (the action of the HMC)
    template<typename ActionType>
    NSL::MCMC::MarkovState<Type> operator()(const NSL::MCMC::MarkovState<Type> & state, ActionType & action) {
        if (proposals_ == 0) {
            return state;
        }
        NSL::MCMC::MarkovState<Type> out = state;
        const NSL::size_t Nx = modes_.size(0);
        for (NSL::size_t n = 0; n < proposals_; ++n) {
            r_.rand();
            const NSL::size_t k = std::min<NSL::size_t>(Nx - 1, NSL::size_t(double(r_[0]) * double(Nx)));
            const double u = double(r_[1]);
            ++stats_->proposed;
            ++stats_->proposedPerMode[k];
            NSL::Configuration<Type> proposal(out.configuration, true);
            proposal[key_] = reflect(out.configuration[key_], k);
            Type S1;
            try {
                S1 = action(proposal);
            } catch (const std::exception & e) {
                ++stats_->failed;
                NSL::Logger::warn("ModeFlip: the action of a proposal failed and the flip is rejected: {}", e.what());
                continue;
            }
            const double logAcceptance = double(NSL::real(out.actionValue - S1));
            if (std::isnan(logAcceptance)) {
                ++stats_->failed;
                NSL::Logger::warn("ModeFlip: the action difference is NaN; the flip is rejected");
                continue;
            }
            if (u <= std::exp(logAcceptance)) {          // false for logAcceptance = -inf
                // swap, not assign: NSL::Tensor's assignment copies into the existing storage, which `out` shares
                // with the caller's state (copies of a configuration are shallow)
                out.configuration.swap(proposal);
                out.actionValue = S1;
                ++stats_->accepted;
                ++stats_->acceptedPerMode[k];
            }
        }
        return out;
    }

    NSL::size_t proposals() const { return proposals_; }
    const torch::Tensor & modes() const { return modes_; }
    const Stats & stats() const { return *stats_; }

    private:
    torch::Tensor modes_;
    NSL::size_t proposals_;
    std::string key_;
    NSL::Tensor<double> r_;
    std::shared_ptr<Stats> stats_;
};

//! The modes of the PPP interaction for mode flips: the eigenvectors of V (Nx x Nx), largest eigenvalue first; the
//! site basis (the identity) if V is a multiple of the identity (Hubbard), where the eigenvectors are not unique.
inline NSL::Tensor<double> interactionModes(const NSL::Tensor<double> & V) {
    const torch::Tensor v = torch::Tensor(V).to(torch::kCPU, torch::kDouble);
    const int64_t n = v.size(0);
    const double d = v[0][0].template item<double>();
    if ((v - d * torch::eye(n, torch::kDouble)).abs().max().template item<double>() <= 1e-12 * std::max(1.0, std::abs(d))) {
        return NSL::Tensor<double>(torch::eye(n, torch::kDouble));
    }
    auto [w, Q] = torch::linalg_eigh(0.5 * (v + v.transpose(0, 1)));
    return NSL::Tensor<double>(Q.flip({1}).contiguous());          // eigh sorts ascending
}

} // namespace NSL::MCMC

#endif // NSL_MODE_FLIP_TPP
