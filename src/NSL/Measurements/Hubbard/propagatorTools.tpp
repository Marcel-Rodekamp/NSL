#ifndef NSL_MEASUREMENTS_HUBBARD_PROPAGATOR_TOOLS_TPP
#define NSL_MEASUREMENTS_HUBBARD_PROPAGATOR_TOOLS_TPP

/*! \file propagatorTools.tpp
 *  Helpers shared by FermionPropagator (charge basis) and FermionPropagatorSpinBasis:
 *  a GPU-safe UDT decomposition, a finiteness guard, the allocator setting for long
 *  measurement runs, the projection of propagators on source vectors, and the selection
 *  and bookkeeping of hopping eigenvectors. The UDT, finiteness guard, allocator setting
 *  and eigenvector selection rule come from Petar Sinilkov's fix/spin-projection branch.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "Tensor.hpp"
#include "LinAlg.hpp"
#include "IO.hpp"
#include "stop.hpp"
#include "logger.hpp"

namespace NSL::Measure::Hubbard::PropagatorTools {

//! UDT decomposition t = U.D.V (NSL::LinAlg::udt) computed with the CPU (LAPACK) QR also when
//! t lives on the GPU; on the CPU this is exactly NSL::LinAlg::udt.
/*!
 *  The matrices factorised by the propagator measurements span |x| ~ 1e-170 .. 1e170 at low
 *  temperature (beta = 90). torch's batched CUDA QR returns non-finite R for finite inputs with
 *  |x| >~ 1e154 = sqrt(DBL_MAX) and for such row/column-graded inputs, while LAPACK stays
 *  finite. With the host QR, GPU and CPU agree to ~1e-10 relative to the largest element
 *  (32 perylene configurations at beta = 90). The host round trip is free for small systems
 *  but slows a GPU run down for Nx >~ 100 (Nx = 112: 1.6x, Nx = 200: 2.1x).
 *  (Measurements by P. Sinilkov.)
 */
template<NSL::Concept::isNumber Type>
std::tuple<NSL::Tensor<Type>,NSL::Tensor<Type>,NSL::Tensor<Type>> udt(const NSL::Tensor<Type> & t){
    const torch::Tensor M = static_cast<torch::Tensor>(t);
    if (!M.is_cuda()) { return NSL::LinAlg::udt(t); }
    auto [Q, D, V] = NSL::LinAlg::udt(NSL::Tensor<Type>(M.cpu()));
    return std::make_tuple( NSL::Tensor<Type>(static_cast<torch::Tensor>(Q).to(M.device())),
                            NSL::Tensor<Type>(static_cast<torch::Tensor>(D).to(M.device())),
                            NSL::Tensor<Type>(static_cast<torch::Tensor>(V).to(M.device())) );
}

//! Throws (so that nothing is written for node) if x has a non-finite entry
inline void requireFinite(const torch::Tensor & x, const std::string & node){
    if (!torch::isfinite(x).all().template item<bool>()) {
        throw std::runtime_error(fmt::format("non-finite propagator values for {}; nothing was written there", node));
    }
}

template<NSL::Concept::isNumber Type>
void requireFinite(const NSL::Tensor<Type> & x, const std::string & node){
    requireFinite(static_cast<torch::Tensor>(x), node);
}

template<NSL::Concept::isNumber Type>
void requireFiniteScalar(const Type & x, const std::string & node){
    bool finite;
    if constexpr (NSL::is_complex<Type>()) {
        finite = std::isfinite(static_cast<double>(x.real())) && std::isfinite(static_cast<double>(x.imag()));
    } else {
        finite = std::isfinite(static_cast<double>(x));
    }
    if (!finite) {
        throw std::runtime_error(fmt::format("non-finite value for {}; nothing was written there", node));
    }
}

//! glibc only (e.g. Jureca): a fixed mmap threshold of 128 KiB for the whole process.
/*!
 *  Every configuration writes many small HDF5 datasets whose long-lived allocations stay in the
 *  heap; with glibc's dynamic threshold the large tensor temporaries are then placed between
 *  them, the freed memory can neither be reused nor returned, and the resident set grew by
 *  several MB per configuration (perylene, Nt = 512) until the node ran out of memory. With a
 *  fixed threshold, allocations >= 128 KiB are mmapped and returned on free. 128 KiB was the
 *  only tested threshold that removes the fragmentation; it costs ~10-14% run time.
 *  (Measurements by P. Sinilkov.) No-op on other C libraries, e.g. macOS.
 */
inline void configureAllocator(){
#if defined(__GLIBC__)
    static const bool configured = (mallopt(M_MMAP_THRESHOLD, 128 * 1024) == 1);
    (void) configured;
#endif
}

/*! true if the stored propagator component at `full` was written completely, so that a rerun may keep it.
 *  NSL::H5IO::write(tensor) writes the data, then the attributes 'shape' and 'type', in this order: a tensor dataset
 *  is complete when it has 'type'. A scalar dataset (Q) has no attributes and counts as complete. A group (one dataset
 *  per pair of wall sources) is complete when it holds `leaves` datasets (if leaves > 0), each complete.
 *  A write interrupted by an error (e.g. a full disk quota) leaves the dataset without its attributes, and its values
 *  are then wrong (seen on JURECA, 2026-11-24): such a component must be measured again, not kept.
 */
inline bool complete(HighFive::File & file, const std::string & full, std::size_t leaves = 0){
    if (file.getObjectType(full) == HighFive::ObjectType::Dataset) {
        HighFive::DataSet ds = file.getDataSet(full);
        if (ds.getElementCount() <= 1 && !ds.hasAttribute("shape")) { return true; }   // a scalar (Q)
        return ds.hasAttribute("type");
    }
    HighFive::Group g = file.getGroup(full);
    const std::vector<std::string> names = g.listObjectNames();
    if (names.empty() || (leaves > 0 && names.size() != leaves)) { return false; }
    for (const std::string & n: names) {
        if (g.getObjectType(n) != HighFive::ObjectType::Dataset || !g.getDataSet(n).hasAttribute("type")) { return false; }
    }
    return true;
}

//! All projections of corr (Nt x Nx x Nx) on the sources w (K x B x Nx) at once:
//! P(t, kSink, sigmaSink, kSrc, sigmaSrc) = sum_xy conj(w(kSink,sigmaSink,x)) corr(t,x,y) w(kSrc,sigmaSrc,y)
template<NSL::Concept::isNumber Type>
torch::Tensor projectAll(const NSL::Tensor<Type> & corr, const NSL::Tensor<Type> & w){
    const torch::Tensor Wt = static_cast<torch::Tensor>(w);
    const int64_t K = Wt.size(0), B = Wt.size(1), Nx = Wt.size(2);
    const torch::Tensor W = Wt.reshape({K*B, Nx});
    const torch::Tensor C = static_cast<torch::Tensor>(corr);               // (Nt, Nx, Nx)
    const torch::Tensor X = torch::matmul(C, W.transpose(0, 1));            // (Nt, Nx, K*B)
    return torch::matmul(W.conj(), X).reshape({C.size(0), K, B, K, B});     // (Nt, K, B, K, B)
}

// =====================================================================================
// Hopping eigenbasis
// =====================================================================================

//! Version of the selection rule recorded with an eigenbasis (4 = P. Sinilkov's rule 4)
inline constexpr int eigenbasisRuleVersion = 4;
inline const std::string eigenbasisRule =
    "eps_n = -(lambda_n + mu) for K v_n = lambda_n v_n; levels = eps within tol; default (Fermi window): "
    "levels HOMO-below .. LUMO+above of the non-interacting filling at mu (a level with |eps| <= tol is "
    "the HOMO), full degenerate subspaces; eigenvalueIndices: the listed ascending-eps indices; one block, "
    "bands by ascending eps";

//! What to select: explicit energy indices, or else the Fermi window HOMO-below .. LUMO+above
struct EigenbasisRequest {
    std::optional<std::vector<int64_t>> energyIndices;
    int64_t below = 1;
    int64_t above = 0;
};

//! The selected hopping eigenvectors and everything recorded about them
struct EigenbasisSelection {
    double mu = 0., tol = 0.;
    std::vector<double> lambda, energy;           //!< all eigenvalues of K and eps_n, eigh order n
    std::vector<int64_t> energyOrder;             //!< eigh index n of energy index i
    std::vector<int64_t> indices, energyIndices;  //!< eigh index and energy index of band j
    std::vector<double> eigenvalues, energies;    //!< lambda and eps of band j
    std::vector<std::string> roles;               //!< "HOMO-1", "HOMO", "LUMO", ... of band j
    int selectionMode = 1;                        //!< 1 = Fermi window, 2 = energy indices
    int64_t below = 1, above = 0;                 //!< Fermi window after clamping
    std::vector<int64_t> requested;               //!< requested energy indices (mode 2)
    int homoAmbiguous = 0;                        //!< 1 if the HOMO is a level with |eps| <= tol
    double homoAmbiguousEps = 0.;                 //!< eps of that level
    torch::Tensor K;                              //!< hopping matrix (CPU, complex double)
    torch::Tensor vectors;                        //!< (Nx, m), column j = vector of band j
};

/*! Selects eigenvectors of the hopping matrix K (rule version 4, by P. Sinilkov).
 *
 *  K v_n = lambda_n v_n (eigh, ascending lambda). The propagator carries e^{delta K} e^{delta mu},
 *  so the single-particle energies are eps_n = -(lambda_n + mu) and a level is occupied for
 *  eps < 0. Levels are groups of eps within tol = 1e-8 max(1, max|eps|) (chained over consecutive
 *  sorted values); the energy index i is the position in ascending eps, ties by n. The
 *  non-interacting filling at mu has its HOMO at the highest level with eps < 0 and its LUMO at
 *  the next level; a level with |eps| <= tol is ambiguous and taken as the HOMO (warning,
 *  homoAmbiguous = 1).
 *   - Fermi window (default): the levels HOMO-below .. LUMO+above with their full degenerate
 *     subspaces; below/above larger than the number of levels are clamped, levels beyond the
 *     spectrum are skipped with a warning.
 *   - energy indices: exactly the listed vectors; duplicates or indices outside [0,Nx) are
 *     errors, a level whose degenerate partners are not all listed gives a warning.
 *  The bands are the selected vectors in ascending eps (ties by n). The eigenvectors do not
 *  depend on mu, only the selection does.
 */
inline EigenbasisSelection selectHoppingEigenbasis(const torch::Tensor & Kin, double mu, const EigenbasisRequest & req){
    EigenbasisSelection s;
    s.mu = mu;
    s.K = Kin.to(torch::kCPU).to(torch::kComplexDouble).contiguous();
    const int64_t Nx = s.K.size(0);

    // real symmetric K gives real eigenvectors
    torch::Tensor evals, evecs;
    if (torch::imag(s.K).abs().max().template item<double>() == 0.) {
        std::tie(evals, evecs) = torch::linalg_eigh(torch::real(s.K).contiguous(), "L");
        evecs = evecs.to(torch::kComplexDouble);
    } else {
        std::tie(evals, evecs) = torch::linalg_eigh(s.K, "L");
    }
    evals = evals.to(torch::kDouble).contiguous();
    s.lambda.assign(evals.data_ptr<double>(), evals.data_ptr<double>() + Nx);
    s.energy.resize(Nx);
    double maxAbs = 0.;
    for (int64_t n = 0; n < Nx; ++n) { s.energy[n] = -(s.lambda[n] + mu); maxAbs = std::max(maxAbs, std::abs(s.energy[n])); }
    s.tol = 1e-8 * std::max(1., maxAbs);

    // levels and the ascending-energy order (ties by n)
    std::vector<int64_t> byEps(Nx), level(Nx);
    std::iota(byEps.begin(), byEps.end(), 0);
    std::stable_sort(byEps.begin(), byEps.end(), [&](int64_t a, int64_t b){ return s.energy[a] < s.energy[b]; });
    int64_t nLevels = 0;
    for (int64_t i = 0; i < Nx; ++i) {
        if (i > 0 && s.energy[byEps[i]] - s.energy[byEps[i-1]] > s.tol) { ++nLevels; }
        level[byEps[i]] = nLevels;
    }
    ++nLevels;
    s.energyOrder.resize(Nx);
    std::iota(s.energyOrder.begin(), s.energyOrder.end(), 0);
    std::sort(s.energyOrder.begin(), s.energyOrder.end(), [&](int64_t a, int64_t b){
        return std::make_pair(level[a], a) < std::make_pair(level[b], b);
    });
    std::vector<int64_t> energyIndex(Nx);
    for (int64_t i = 0; i < Nx; ++i) { energyIndex[s.energyOrder[i]] = i; }

    // non-interacting filling: HOMO level h (-1 if none), LUMO level h+1
    int64_t h = -1, zeroLevel = -1;
    for (int64_t n = 0; n < Nx; ++n) {
        if (std::abs(s.energy[n]) <= s.tol) { zeroLevel = level[n]; s.homoAmbiguousEps = s.energy[n]; }
        else if (s.energy[n] < 0) { h = std::max(h, level[n]); }
    }
    if (zeroLevel >= 0) {
        NSL::Logger::warn("Hopping eigenbasis: the level at eps = 0 (|eps| <= {:.1e}) is ambiguous at mu = {}; it is taken as the non-interacting HOMO", s.tol, mu);
        h = zeroLevel;
        s.homoAmbiguous = 1;
    } else {
        s.homoAmbiguousEps = 0.;
    }
    const int64_t l = h + 1;
    auto role = [&](int64_t c){
        if (c <= h) { return c == h ? std::string("HOMO") : fmt::format("HOMO-{}", h - c); }
        return c == l ? std::string("LUMO") : fmt::format("LUMO+{}", c - l);
    };

    std::set<int64_t> selected;
    if (req.energyIndices) {
        s.selectionMode = 2;
        for (int64_t r: *req.energyIndices) {
            if (r < 0 || r >= Nx) {
                throw std::runtime_error(fmt::format("eigenvalueIndices: {} is not an energy index in [0,{})", r, Nx));
            }
            if (!selected.insert(s.energyOrder[r]).second) {
                throw std::runtime_error(fmt::format("eigenvalueIndices: index {} is listed twice", r));
            }
            s.requested.push_back(r);
        }
        for (int64_t n: selected) {
            std::string missing;
            for (int64_t m = 0; m < Nx; ++m) {
                if (!selected.contains(m) && level[m] == level[n]) { missing += fmt::format(" {}", energyIndex[m]); }
            }
            if (!missing.empty()) {
                NSL::Logger::warn("Hopping eigenbasis: energy index {} (eps = {:.6f}) is degenerate with the unlisted energy indices{}; the selection depends on the eigensolver's basis of that subspace", energyIndex[n], s.energy[n], missing);
            }
        }
    } else {
        s.selectionMode = 1;
        if (req.below < 0 || req.above < 0) {
            throw std::runtime_error(fmt::format("fermiWindow below and above must be >= 0 (got {}, {})", req.below, req.above));
        }
        s.below = std::min<int64_t>(req.below, nLevels);
        s.above = std::min<int64_t>(req.above, nLevels);
        if (s.below != req.below || s.above != req.above) {
            NSL::Logger::info("Hopping eigenbasis: Fermi window HOMO-{} .. LUMO+{} clamped to the {} distinct levels", req.below, req.above, nLevels);
        }
        int64_t missingBelow = 0, missingAbove = 0;
        for (int64_t c = h - s.below; c <= l + s.above; ++c) {
            if (c < 0) { ++missingBelow; continue; }
            if (c >= nLevels) { ++missingAbove; continue; }
            for (int64_t n = 0; n < Nx; ++n) { if (level[n] == c) { selected.insert(n); } }
        }
        if (missingBelow > 0 || missingAbove > 0) {
            NSL::Logger::warn("Hopping eigenbasis: the Fermi window HOMO-{} .. LUMO+{} extends {} level(s) below and {} level(s) above the spectrum at mu = {}; these are skipped", s.below, s.above, missingBelow, missingAbove, mu);
        }
    }
    if (selected.empty()) { throw std::runtime_error("the hopping eigenbasis selection is empty"); }

    s.indices.assign(selected.begin(), selected.end());
    std::sort(s.indices.begin(), s.indices.end(), [&](int64_t a, int64_t b){ return energyIndex[a] < energyIndex[b]; });
    for (int64_t n: s.indices) {
        s.energyIndices.push_back(energyIndex[n]);
        s.eigenvalues.push_back(s.lambda[n]);
        s.energies.push_back(s.energy[n]);
        s.roles.push_back(role(level[n]));
    }
    s.vectors = evecs.index_select(1, torch::tensor(s.indices, torch::kLong)).contiguous();

    std::string desc;
    for (size_t j = 0; j < s.indices.size(); ++j) {
        desc += fmt::format("{}band {}: n={} energy index {} lambda={:.6f} eps={:.6f} {}", j ? "; " : "", j, s.indices[j], s.energyIndices[j], s.eigenvalues[j], s.energies[j], s.roles[j]);
    }
    NSL::Logger::info("Hopping eigenbasis ({}, mu = {}): {} of {} eigenvectors: {}",
        s.selectionMode == 2 ? std::string("eigenvalueIndices") : fmt::format("Fermi window HOMO-{} .. LUMO+{}", s.below, s.above),
        mu, s.indices.size(), Nx, desc);
    return s;
}

/*! Records the selection in node (<name>/Meta/propagator/eigenbasis) or checks it against a stored
 *  one, and returns the eigenvectors to project on, (Nx, m) complex double on the CPU.
 *
 *  A stored eigenbasis is reused, including its eigenvectors (so that signs, phases and the basis
 *  of degenerate subspaces stay the same for all configurations), if rule version, mu and the
 *  selected eigh indices agree and its vectors are orthonormal eigenvectors of K with the stored
 *  eigenvalues. Different selection parameters that select the same vectors are only logged.
 *  Otherwise it is an error, unless overwrite is set: then the record is replaced (all eigenbasis
 *  results are recomputed in that case anyway).
 *
 *  Contents: ruleVersion (attribute "rule"), mu, selectionMode (1 = fermiWindow, 2 =
 *  eigenvalueIndices; attribute "mode"), fermiWindowBelow/Above (mode 1, after clamping),
 *  requestedIndices (mode 2), homoAmbiguous (+ homoAmbiguousEps), lambda and energy (all Nx, eigh
 *  order), energyOrder (eigh index of energy index i), and per band j: indices (eigh index),
 *  energyIndices, eigenvalues, energies, roles, and eigenvectors (Nx, m), column j.
 */
template<NSL::Concept::isNumber Type>
torch::Tensor recordEigenbasis(NSL::H5IO & h5, const std::string & node, const EigenbasisSelection & s, bool overwrite){
    const int64_t Nx = s.K.size(0), m = s.indices.size();
    auto readLong = [&](const std::string & n){
        NSL::Tensor<int> t; h5.read(t, n);
        const torch::Tensor tt = static_cast<torch::Tensor>(t).to(torch::kCPU).to(torch::kLong).contiguous();
        return std::vector<int64_t>(tt.data_ptr<int64_t>(), tt.data_ptr<int64_t>() + tt.numel());
    };

    if (h5.exist(node)) {
        std::string why;
        int rule = 0;
        double mu = 0.;
        if (!h5.exist(node + "/ruleVersion")) { why = "it has no rule version"; }
        else {
            h5.read(rule, node + "/ruleVersion");
            h5.read(mu, node + "/mu");
            if (rule != eigenbasisRuleVersion) { why = fmt::format("rule version {} (this run: {})", rule, eigenbasisRuleVersion); }
            else if (std::abs(mu - s.mu) > 1e-12 * std::max(1., std::abs(s.mu))) { why = fmt::format("mu = {} (this run: {})", mu, s.mu); }
            else if (readLong(node + "/indices") != s.indices) { why = "it selects different eigenvectors"; }
        }
        if (why.empty()) {
            NSL::Tensor<Type> Vs; NSL::Tensor<double> ls;
            h5.read(Vs, node + "/eigenvectors");
            h5.read(ls, node + "/eigenvalues");
            const torch::Tensor V = static_cast<torch::Tensor>(Vs).to(torch::kCPU).to(torch::kComplexDouble);
            const torch::Tensor lam = static_cast<torch::Tensor>(ls).to(torch::kCPU).to(torch::kDouble);
            const bool ok = V.dim() == 2 && V.size(0) == Nx && V.size(1) == m && lam.numel() == m
                && (lam - torch::tensor(s.eigenvalues, torch::kDouble)).abs().max().template item<double>() <= s.tol
                && (torch::matmul(s.K, V) - V * lam.to(torch::kComplexDouble)).abs().max().template item<double>() <= s.tol
                && (torch::matmul(V.conj().transpose(0,1), V) - torch::eye(m, torch::kComplexDouble)).abs().max().template item<double>() <= 1e-8;
            if (ok) {
                int mode = 0;
                h5.read(mode, node + "/selectionMode");
                if (mode != s.selectionMode) {
                    NSL::Logger::info("Hopping eigenbasis: the stored selection used {}, this run {}; both select the same eigenvectors",
                        mode == 2 ? "eigenvalueIndices" : "the Fermi window", s.selectionMode == 2 ? "eigenvalueIndices" : "the Fermi window");
                }
                NSL::Logger::info("Hopping eigenbasis: using the eigenvectors stored in {}", node);
                return V;
            }
            why = "its eigenvectors do not match the hopping matrix";
        }
        if (!overwrite) {
            throw std::runtime_error(fmt::format(
                "the hopping eigenbasis stored in {} differs from this run: {}. Results projected on different "
                "eigenvectors must not be mixed; use the same selection, another file, or --overwrite to measure "
                "all eigenbasis results again.", node, why));
        }
        NSL::Logger::info("Hopping eigenbasis: replacing the record in {} ({}) because of --overwrite", node, why);
        h5.deleteData(node);
    }

    // the record is one write section: a SIGTERM waits until it is complete (NSL::Stop)
    NSL::Stop::WriteGuard writing;
    HighFive::File & file = h5.getFile();
    auto attr = [&](const std::string & n, const std::string & name, const std::string & value){
        file.getDataSet(n).createAttribute<std::string>(name, HighFive::DataSpace::From(value)).write(value);
    };
    auto dbl = [](const std::vector<double> & v){ return NSL::Tensor<double>(torch::tensor(v, torch::kDouble)); };
    auto itg = [](const std::vector<int64_t> & v){ return NSL::Tensor<int>(torch::tensor(v, torch::kLong).to(torch::kInt)); };

    h5.write(eigenbasisRuleVersion, node + "/ruleVersion");  attr(node + "/ruleVersion", "rule", eigenbasisRule);
    h5.write(s.mu, node + "/mu");
    h5.write(s.selectionMode, node + "/selectionMode");
    attr(node + "/selectionMode", "mode", s.selectionMode == 2 ? "eigenvalueIndices" : "fermiWindow");
    if (s.selectionMode == 2) {
        NSL::Tensor<int> r = itg(s.requested); h5.write(r, node + "/requestedIndices");
    } else {
        h5.write(int(s.below), node + "/fermiWindowBelow");
        h5.write(int(s.above), node + "/fermiWindowAbove");
    }
    h5.write(s.homoAmbiguous, node + "/homoAmbiguous");
    if (s.homoAmbiguous) { h5.write(s.homoAmbiguousEps, node + "/homoAmbiguousEps"); }
    NSL::Tensor<double> lambda = dbl(s.lambda), energy = dbl(s.energy), eigenvalues = dbl(s.eigenvalues), energies = dbl(s.energies);
    NSL::Tensor<int> energyOrder = itg(s.energyOrder), indices = itg(s.indices), energyIndices = itg(s.energyIndices);
    NSL::Tensor<Type> vectors(s.vectors.clone());
    h5.write(lambda, node + "/lambda");
    h5.write(energy, node + "/energy");
    h5.write(energyOrder, node + "/energyOrder");
    h5.write(indices, node + "/indices");
    h5.write(energyIndices, node + "/energyIndices");
    h5.write(eigenvalues, node + "/eigenvalues");
    h5.write(energies, node + "/energies");
    file.createDataSet<std::string>(node + "/roles", HighFive::DataSpace::From(s.roles)).write(s.roles);
    h5.write(vectors, node + "/eigenvectors");
    file.flush();
    return s.vectors;
}

} // namespace NSL::Measure::Hubbard::PropagatorTools

#endif // NSL_MEASUREMENTS_HUBBARD_PROPAGATOR_TOOLS_TPP
