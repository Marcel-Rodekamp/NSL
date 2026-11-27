#ifndef NSL_REDUCED_OBSERVABLES_TPP
#define NSL_REDUCED_OBSERVABLES_TPP

#include "fermionPropagator.tpp"
#include "stop.hpp"
#include <highfive/H5File.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace NSL::Measure::Hubbard {

/*! Observables of a charge-basis (HubbardExp) ensemble, measured one configuration at a time from the auxiliary
 *  field, so that the propagators are never written: the content of the stage-1 file of
 *  Executables/PPP/python/ppp_reduce.py (single-particle correlators by orbital class, equal-time G, <s s> and
 *  <q q>, Monte Carlo history, determinant sign, fermion action at phi, bridge-link weights) and, optionally, of the
 *  stage-2 file of ppp_reduce2.py (exciton and pp correlator matrices of the orbitals near the Fermi level),
 *  computed in memory.
 *
 *  The output files are written first by `ppp_reduce.py --skeleton` and `ppp_reduce2.py --skeleton` (attributes,
 *  orbitals and every per-configuration and per-bin dataset with length 0, extendible), so that the orbital basis,
 *  the operator bases and the file layout are those of the Python reduction. This class reads from them what it
 *  needs (sources, bin size, precision, orbitals and their classes, probe slices, the interaction V, the bridge link,
 *  the operator pairs), writes noninteracting/ of stage 1 once, and appends one bin of `bin` consecutive
 *  configurations at a time: first the datasets, then the attributes n_configs, n_bins and checks/attrs, then a
 *  flush. A file therefore holds complete bins only; configurations of an incomplete bin are lost when the run stops
 *  and are measured again from the stored fields: next() is the configuration id expected next, the first one that
 *  is missing in either file (a bin that only one of them lacks is written to that one only, e.g. after a stop
 *  between the two writes, or when stage 2 is added to an ensemble measured before). Datasets longer than n_bins
 *  (an interrupted append) are cut back when a file is opened.
 *
 *  Configurations are identified by their id in the chain (markovChain/<id>), which must be 0, 1, 2, ... in order.
 *  Bridge-link chains: every per-configuration quantity is multiplied by w = 1/Lambda before binning, and the bins
 *  of w are stored in `weights` of both files, as in the Python reduction. mu = 0 on a bipartite lattice is required
 *  for the sign and the fermion action (config/S_fermion_phi exists only then) and for stage 2.
 */
template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
class ReducedObservables {
    public:
    using Propagator = FermionPropagator<Type, LatticeType, NSL::FermionMatrix::HubbardExp<Type, LatticeType>>;

    /*! lattice, params as for FermionPropagator (name, beta, Nt, Nx, mu, device, Number Time Sources); chain: the
     *  ensemble file (only its Meta/bridge record is read); stage1, stage2: the files written by
     *  ppp_reduce.py --skeleton and ppp_reduce2.py --skeleton (or ones this class appended to before); stage2 = "":
     *  stage 1 only. */
    ReducedObservables(LatticeType & lattice, NSL::Parameter params, NSL::H5IO & chain, const std::string & stage1,
                       const std::string & stage2 = "");

    //! the configuration id expected by the next add()
    NSL::size_t next() const { return complete_() + pending_; }
    //! complete bins in both files, configurations per bin
    NSL::size_t bins() const { return complete_() / B_; }
    NSL::size_t binSize() const { return B_; }
    bool stage2() const { return bool(out2_); }

    /*! Measures configuration cfgID (must equal next()) with field phi (Nt, Nx) and the values stored with it by the
     *  HMC (action S, acceptance probability, Markov time); writes a bin when it is complete. */
    void add(NSL::size_t cfgID, const NSL::Tensor<Type> & phi, Type S, double acceptanceProbability, double markovTime);

    //! largest deviation from the particle-hole relation found on the checked configurations (ids < checkFirst_)
    double holeRelationDeviation() const { return holeDev_; }
    //! largest relative deviation of the stage-2 pair contractions from the general ones (ids < checkFirst_)
    double contractionDeviation() const { return std::max(excDev_, ppDev_); }

    private:
    // --- setup ---
    template<typename T> static T attr_(HighFive::File & f, const std::string & name) {
        T v; f.getAttribute(name).read(v); return v;
    }
    //! the bridge slice of a file attribute or a chain dataset: a slice number, or allSlices for the string "all"
    static constexpr NSL::size_t allSlices = std::numeric_limits<NSL::size_t>::max();
    template<typename H> static NSL::size_t bridgeSlice_(const H & obj, const std::string & where) {
        if (obj.getDataType().getClass() == HighFive::DataTypeClass::String) {
            std::string name;
            obj.read(name);
            if (name != "all") {
                throw std::runtime_error(fmt::format("{}: bridge slice \"{}\"; expected a number or \"all\"", where, name));
            }
            return allSlices;
        }
        int64_t v;
        obj.read(v);
        return NSL::size_t(v);
    }
    static std::string sliceName_(NSL::size_t slice) {
        return slice == allSlices ? std::string("all") : std::to_string(slice);
    }
    torch::Tensor readMatrix_(HighFive::File & f, const std::string & path) const;   // real 2D dataset -> double tensor
    std::vector<std::array<int64_t, 2>> readPairs_(HighFive::File & f, const std::string & path) const;
    void openStage2_(const std::string & stage2, NSL::size_t Nt, NSL::size_t Nx);
    void cutBack_(HighFive::File & f, const std::vector<std::string> & paths, NSL::size_t length) const;
    void writeNonInteracting_();
    NSL::size_t complete_() const { return out2_ ? std::min(nConfigs_, nConfigs2_) : nConfigs_; }

    // --- per configuration ---
    struct OneBody { torch::Tensor F, B, Gtt; };
    OneBody oneBody_(const typename Propagator::Spatial & P, const std::vector<NSL::size_t> & sources) const;
    torch::Tensor project_(const torch::Tensor & M) const {              // U^T M U, (Nt, Nx, Nx)
        return torch::matmul(torch::matmul(Ut_, M), U_);
    }
    torch::Tensor projectWindow_(const torch::Tensor & M) const {        // Uw^T M Uw, (Nt, w, w)
        return torch::matmul(torch::matmul(UwT_, M), Uw_);
    }
    static torch::Tensor block_(const torch::Tensor & M, const torch::Tensor & rows, const torch::Tensor & cols) {
        return M.index_select(1, rows).index_select(2, cols);
    }
    // (E, ss, qq) of ppp_wick.equal_time, averaged over the time slices
    std::array<torch::Tensor, 3> equalTime_(const torch::Tensor & Gtt) const;
    double holeRelation_(const typename Propagator::Spatial & P, const typename Propagator::Spatial & H) const;
    void stage2Config_(const typename Propagator::Spatial & P, const torch::Tensor & Gtt, double w);
    void checkContractions_(const typename Propagator::Spatial & P, const torch::Tensor & Gtt);

    // --- output ---
    void writeBin_();
    void writeStage1_();
    void writeStage2_();
    template<typename T> static void append_(HighFive::File & f, const std::string & path, const T * data, NSL::size_t rows, NSL::size_t at);
    static void appendComplex_(HighFive::File & f, bool single, const std::string & path, const torch::Tensor & t, NSL::size_t rows, NSL::size_t at);
    static void appendReal_(HighFive::File & f, bool single, const std::string & path, const torch::Tensor & t, NSL::size_t rows, NSL::size_t at);

    Propagator invM_;
    HighFive::File out_;
    std::unique_ptr<HighFive::File> out2_;
    NSL::size_t Nt_, Nx_, B_, nBins_, nConfigs_, nBins2_ = 0, nConfigs2_ = 0, pending_ = 0, binStart_ = 0;
    double delta_;
    std::vector<NSL::size_t> sources_;
    bool bridge_ = false, doSign_ = false, single_ = false, single2_ = false;
    double theta_ = 0;
    NSL::size_t slice_ = 0;
    torch::Tensor U_, Ut_, Ainv_, QV_, lamV_, probe_, back_, probePositive_;
    std::vector<std::string> classNames_;
    std::map<std::string, torch::Tensor> classIdx_;
    torch::TensorOptions cplx_ = torch::TensorOptions().dtype(torch::kComplexDouble);
    torch::TensorOptions real_ = torch::TensorOptions().dtype(torch::kDouble);
    torch::TensorOptions long_ = torch::TensorOptions().dtype(torch::kLong);
    static constexpr NSL::size_t checkFirst_ = 2;   // configurations checked (particle-hole relation, contractions)

    // stage 2: window orbitals Uw (Nx, w) and the operator pairs as window positions, by sector
    torch::Tensor Uw_, UwT_;
    std::map<int64_t, int64_t> wpos_;                // orbital index -> position in the window
    struct ExcitonSector { std::string name; torch::Tensor c, v; };
    struct PairSector { std::string path; double sign; torch::Tensor a, b, al2; };
    std::vector<ExcitonSector> exc_;
    std::vector<PairSector> pp_;
    std::array<int64_t, 2> checkExc_{}, checkPP_{};   // orbital pairs of the contraction check

    // the current bin: weighted sums and per-configuration buffers
    std::map<std::string, torch::Tensor> accF_, accB_, accX_, accD_, accEx_, accPP_;
    std::array<torch::Tensor, 3> accEq_;
    double wsum_ = 0;
    std::vector<int64_t> ids_;
    std::vector<std::complex<double>> S_;
    std::vector<double> P_, T_, Sg_, res_, SF_, lam_;
    std::vector<signed char> sign_;
    std::vector<torch::Tensor> modes_;
    std::map<std::string, std::vector<torch::Tensor>> probes_;

    // checks (running maxima, as in checks/attrs)
    double holeDev_ = 0, signRes_ = 0, reNDev_ = 0, sfDev_ = 0, excDev_ = 0, ppDev_ = 0;
    int64_t holeConfigs_ = 0, checkedConfigs_ = 0;
};

// ------------------------------------------------------------------------------------------------------------------

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
ReducedObservables<Type, LatticeType>::ReducedObservables(LatticeType & lattice, NSL::Parameter params,
                                                          NSL::H5IO & chain, const std::string & stage1,
                                                          const std::string & stage2):
    invM_(lattice, params, chain),
    out_(stage1, HighFive::File::ReadWrite)
{
    invM_.hfm_.stabilityMethod = "QR";
    Nt_ = params["Nt"].template to<NSL::size_t>();
    Nx_ = params["Nx"].template to<NSL::size_t>();
    const std::string where = fmt::format("{} (written by ppp_reduce.py --skeleton)", stage1);

    if (NSL::size_t(attr_<int64_t>(out_, "Nt")) != Nt_ || NSL::size_t(attr_<int64_t>(out_, "Nx")) != Nx_) {
        throw std::runtime_error(fmt::format("{}: Nt = {}, Nx = {}, but the run has Nt = {}, Nx = {}",
            where, attr_<int64_t>(out_, "Nt"), attr_<int64_t>(out_, "Nx"), Nt_, Nx_));
    }
    const double beta = params["beta"].template to<double>();
    if (std::abs(attr_<double>(out_, "beta") - beta) > 1e-12 * std::max(1.0, beta)) {
        throw std::runtime_error(fmt::format("{}: beta = {}, but the run has beta = {}", where, attr_<double>(out_, "beta"), beta));
    }
    delta_ = attr_<double>(out_, "delta");
    B_ = NSL::size_t(attr_<int64_t>(out_, "bin"));
    nBins_ = NSL::size_t(attr_<int64_t>(out_, "n_bins"));
    nConfigs_ = NSL::size_t(attr_<int64_t>(out_, "n_configs"));
    if (B_ == 0 || nConfigs_ != nBins_ * B_) {
        throw std::runtime_error(fmt::format("{}: n_configs = {} is not n_bins = {} times bin = {}", where, nConfigs_, nBins_, B_));
    }

    // sources: those of the file, which must be those of the run
    for (int64_t s: attr_<std::vector<int64_t>>(out_, "sources")) { sources_.push_back(NSL::size_t(s)); }
    const std::vector<NSL::size_t> expected = Propagator::sourceTimes(Nt_, params["Number Time Sources"].template to<NSL::size_t>());
    if (sources_ != expected) {
        auto list = [](const std::vector<NSL::size_t> & v){
            std::string out;
            for (NSL::size_t x: v) { out += (out.empty() ? "" : ", ") + std::to_string(x); }
            return "[" + out + "]";
        };
        throw std::runtime_error(fmt::format("{}: sources {} differ from those of `Number Time Sources` of the run, {}",
            where, list(sources_), list(expected)));
    }

    // bridge link: the record of the chain (if any) must agree with the file
    bridge_ = attr_<int64_t>(out_, "bridge") != 0;
    if (bridge_) { theta_ = attr_<double>(out_, "bridge_theta"); slice_ = bridgeSlice_(out_.getAttribute("bridge_slice"), where); }
    const std::string meta = params["name"].template to<std::string>() + "/Meta/bridge";
    if (chain.exist(meta + "/theta")) {
        double theta;
        chain.read(theta, meta + "/theta");
        const NSL::size_t slice = bridgeSlice_(chain.getFile().getDataSet(meta + "/slice"), meta + "/slice");
        if (!bridge_ || std::abs(theta - theta_) > 1e-12 || slice != slice_) {
            throw std::runtime_error(fmt::format("the chain was generated with the bridge link theta = {}, slice {}, but {} "
                "has {}", theta, sliceName_(slice), where,
                bridge_ ? fmt::format("theta = {}, slice {}", theta_, sliceName_(slice_)) : std::string("none")));
        }
    } else if (bridge_) {
        NSL::Logger::warn("{} has the bridge link, but the chain has no record of it ({}); using theta = {}, slice {}",
            stage1, meta, theta_, sliceName_(slice_));
    }
    if (bridge_ && slice_ != allSlices && slice_ >= Nt_) {
        throw std::runtime_error(fmt::format("{}: bridge slice {} outside 0..{}", where, slice_, Nt_ - 1));
    }
    doSign_ = out_.exist("config/S_fermion_phi");

    // orbitals, classes, probe slices
    const torch::Tensor U = readMatrix_(out_, "orbitals/U");
    U_ = U.to(torch::kComplexDouble);
    Ut_ = U.transpose(0, 1).contiguous().to(torch::kComplexDouble);
    HighFive::Group onebody = out_.getGroup("onebody");
    for (const std::string & c: onebody.listObjectNames()) {
        std::vector<int64_t> idx;
        onebody.getDataSet(c + "/orbitals").read(idx);
        classNames_.push_back(c);
        classIdx_[c] = torch::tensor(idx, long_);
    }
    std::vector<int64_t> probe;
    onebody.getAttribute("probe_slices").read(probe);
    probe_ = torch::tensor(probe, long_);
    back_ = (int64_t(Nt_) - probe_).remainder(int64_t(Nt_));
    probePositive_ = (probe_ > 0).unsqueeze(1);
    single_ = out_.getDataSet("onebody/" + classNames_.at(0) + "/forward").getDataType().getSize() == 2 * sizeof(float);

    // gauge action (delta V)^-1 and the eigenvectors of V (sorted by descending eigenvalue, as stored)
    Ainv_ = torch::inverse(delta_ * readMatrix_(out_, "lattice/V"));
    QV_ = readMatrix_(out_, "lattice/V_eigenvectors");
    std::vector<double> lamV;
    out_.getDataSet("lattice/V_eigenvalues").read(lamV);
    lamV_ = torch::tensor(lamV, real_);

    // checks of the earlier bins
    HighFive::Group checks = out_.getGroup("checks");
    checks.getAttribute("hole_relation_max_dev").read(holeDev_);
    checks.getAttribute("hole_relation_configs").read(holeConfigs_);
    if (doSign_) { checks.getAttribute("detsign_max_residual").read(signRes_); }
    if (bridge_) { checks.getAttribute("bridge_re_n_max_dev").read(reNDev_); }
    if (doSign_ && !bridge_) { checks.getAttribute("S_fermion_phi_max_dev").read(sfDev_); }

    std::vector<std::string> perConfig, perBin = {"weights", "equaltime/G", "equaltime/ss", "equaltime/qq"};
    for (const std::string k: {"id", "S", "acceptanceProbability", "markovTime", "S_gauge", "field_modes", "detsign",
                               "detsign_residual", "S_fermion_phi", "bridge_lambda"}) {
        perConfig.push_back("config/" + k);
    }
    for (const std::string & c: classNames_) {
        perBin.push_back("onebody/" + c + "/forward");
        perBin.push_back("onebody/" + c + "/backward");
        perConfig.push_back("onebody/" + c + "/addition_probe");
    }
    cutBack_(out_, perConfig, nConfigs_);
    cutBack_(out_, perBin, nBins_);
    if (!out_.exist("noninteracting")) { writeNonInteracting_(); }

    if (!stage2.empty()) { openStage2_(stage2, Nt_, Nx_); }
    NSL::Logger::info("Reduced observables in {}{}: {} and {} complete bins of {} configurations; next configuration {}",
        stage1, out2_ ? " and " + stage2 : std::string(""), nBins_, out2_ ? nBins2_ : nBins_, B_, next());
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void ReducedObservables<Type, LatticeType>::openStage2_(const std::string & stage2, NSL::size_t Nt, NSL::size_t Nx){
    out2_ = std::make_unique<HighFive::File>(stage2, HighFive::File::ReadWrite);
    HighFive::File & f = *out2_;
    const std::string where = fmt::format("{} (written by ppp_reduce2.py --skeleton)", stage2);
    std::vector<NSL::size_t> sources;
    for (int64_t s: attr_<std::vector<int64_t>>(f, "sources")) { sources.push_back(NSL::size_t(s)); }
    if (NSL::size_t(attr_<int64_t>(f, "Nt")) != Nt || NSL::size_t(attr_<int64_t>(f, "Nx")) != Nx
        || NSL::size_t(attr_<int64_t>(f, "bin")) != B_ || sources != sources_
        || (attr_<int64_t>(f, "bridge") != 0) != bridge_) {
        throw std::runtime_error(fmt::format("{}: Nt, Nx, bin, sources or the bridge link differ from the stage-1 file", where));
    }
    if (bridge_ && (std::abs(attr_<double>(f, "bridge_theta") - theta_) > 1e-12
                    || bridgeSlice_(f.getAttribute("bridge_slice"), where) != slice_)) {
        throw std::runtime_error(fmt::format("{}: the bridge link differs from the stage-1 file", where));
    }
    nBins2_ = NSL::size_t(attr_<int64_t>(f, "n_bins"));
    nConfigs2_ = NSL::size_t(attr_<int64_t>(f, "n_configs"));
    if (nConfigs2_ != nBins2_ * B_) {
        throw std::runtime_error(fmt::format("{}: n_configs = {} is not n_bins = {} times bin = {}", where, nConfigs2_, nBins2_, B_));
    }
    // the same orbitals (up to rounding: two runs of the Python setup may differ in the last digits)
    const double dU = (readMatrix_(f, "orbitals/U") - torch::real(U_)).abs().max().template item<double>();
    if (dU > 1e-10) {
        throw std::runtime_error(fmt::format("{}: orbitals/U differs from the stage-1 file by {:.1e}", where, dU));
    }

    // window orbitals and the operator pairs (orbital indices -> window positions)
    std::vector<int64_t> window;
    f.getDataSet("orbitals/window").read(window);
    for (std::size_t k = 0; k < window.size(); ++k) { wpos_[window[k]] = int64_t(k); }
    const torch::Tensor w = torch::tensor(window, long_);
    Uw_ = U_.index_select(1, w).contiguous();
    UwT_ = Uw_.transpose(0, 1).contiguous();
    auto positions = [&](const std::vector<std::array<int64_t, 2>> & pairs, int k){
        std::vector<int64_t> out;
        for (const auto & p: pairs) { out.push_back(wpos_.at(p[k])); }
        return torch::tensor(out, long_);
    };
    for (const std::string & s: f.getGroup("exciton").listObjectNames()) {
        const auto pairs = readPairs_(f, "exciton/" + s + "/pairs");
        exc_.push_back({s, positions(pairs, 0), positions(pairs, 1)});
    }
    for (const auto & [kind, sign]: std::vector<std::pair<std::string, double>>{{"pp_singlet", +1.}, {"pp_triplet", -1.}}) {
        if (!f.exist(kind)) { continue; }
        for (const std::string & s: f.getGroup(kind).listObjectNames()) {
            const auto pairs = readPairs_(f, kind + "/" + s + "/pairs");
            std::vector<double> al;
            for (const auto & p: pairs) { al.push_back(p[0] == p[1] ? 0.5 : 1 / std::sqrt(2.0)); }
            const torch::Tensor a = torch::tensor(al, real_);
            // 2 al_i al_j, in the order of ppp_wick.pp_pairs
            pp_.push_back({kind + "/" + s, sign, positions(pairs, 0), positions(pairs, 1),
                           (2 * a.unsqueeze(1) * a.unsqueeze(0)).to(torch::kComplexDouble)});
        }
    }
    single2_ = exc_.empty() ? single_
        : f.getDataSet("exciton/" + exc_.at(0).name + "/connected").getDataType().getSize() == 2 * sizeof(float);

    HighFive::Group checks = f.getGroup("checks");
    checks.getAttribute("exciton_max_rel_dev").read(excDev_);
    checks.getAttribute("pp_max_rel_dev").read(ppDev_);
    checks.getAttribute("checked_configs").read(checkedConfigs_);
    std::vector<int64_t> pe, pp;
    checks.getAttribute("exciton_check_pair").read(pe);
    checks.getAttribute("pp_check_pair").read(pp);
    checkExc_ = {pe.at(0), pe.at(1)};
    checkPP_ = {pp.at(0), pp.at(1)};

    std::vector<std::string> perBin = {"config/first_id", "weights"};
    for (const auto & e: exc_) {
        for (const std::string k: {"connected", "disconnected", "expectation"}) { perBin.push_back("exciton/" + e.name + "/" + k); }
    }
    for (const auto & p: pp_) { perBin.push_back(p.path + "/correlator"); }
    cutBack_(f, perBin, nBins2_);
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
torch::Tensor ReducedObservables<Type, LatticeType>::readMatrix_(HighFive::File & f, const std::string & path) const {
    std::vector<std::vector<double>> m;
    f.getDataSet(path).read(m);
    torch::Tensor t = torch::empty({int64_t(m.size()), int64_t(m.at(0).size())}, real_);
    for (std::size_t i = 0; i < m.size(); ++i) {
        for (std::size_t j = 0; j < m[i].size(); ++j) { t[i][j] = m[i][j]; }
    }
    return t;
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
std::vector<std::array<int64_t, 2>> ReducedObservables<Type, LatticeType>::readPairs_(HighFive::File & f, const std::string & path) const {
    std::vector<std::vector<int64_t>> m;
    f.getDataSet(path).read(m);
    std::vector<std::array<int64_t, 2>> out;
    for (const auto & row: m) { out.push_back({row.at(0), row.at(1)}); }
    return out;
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void ReducedObservables<Type, LatticeType>::cutBack_(HighFive::File & f, const std::vector<std::string> & paths, NSL::size_t length) const {
    for (const std::string & path: paths) {
        if (!f.exist(path)) { continue; }
        HighFive::DataSet ds = f.getDataSet(path);
        std::vector<std::size_t> dims = ds.getDimensions();
        if (dims.at(0) > std::size_t(length)) {
            NSL::Logger::warn("{}: {} entries beyond the {} of the complete bins (an interrupted append) are removed",
                path, dims[0] - length, length);
            dims[0] = length;
            ds.resize(dims);
        }
    }
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
typename ReducedObservables<Type, LatticeType>::OneBody
ReducedObservables<Type, LatticeType>::oneBody_(const typename Propagator::Spatial & P, const std::vector<NSL::size_t> & sources) const {
    // source averages of the forward G(s + t, s) and backward G(s, s + t), antiperiodic in time (ppp_reduce.onebody)
    const int64_t Nt = int64_t(Nt_);
    const torch::Tensor t = torch::arange(Nt, long_);
    torch::Tensor F = torch::zeros({Nt, int64_t(Nx_), int64_t(Nx_)}, cplx_);
    torch::Tensor B = torch::zeros_like(F);
    for (std::size_t i = 0; i < sources.size(); ++i) {
        const int64_t s = int64_t(sources[i]);
        const torch::Tensor idx = (t + s).remainder(Nt);
        const torch::Tensor sign = (1.0 - 2.0 * (t + s >= Nt).to(torch::kDouble)).to(torch::kComplexDouble).view({Nt, 1, 1});
        F = F + sign * torch::Tensor(NSL::Tensor<Type>(P.column[i], false)).index_select(0, idx);
        B = B + sign * torch::Tensor(NSL::Tensor<Type>(P.row[i], false)).index_select(0, idx);
    }
    const double n = double(sources.size());
    return {F / n, B / n, torch::Tensor(NSL::Tensor<Type>(P.diagonal, false))};
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
std::array<torch::Tensor, 3> ReducedObservables<Type, LatticeType>::equalTime_(const torch::Tensor & Gtt) const {
    // ppp_wick.equal_time: E = 1 - G^T, ss = 2 E G (elementwise), qq = q q^T + ss with q = 2 diag(E) - 1
    const torch::Tensor E = torch::eye(int64_t(Nx_), cplx_) - Gtt.transpose(1, 2);
    const torch::Tensor exch = 2.0 * E * Gtt;
    const torch::Tensor q = 2.0 * torch::diagonal(E, 0, 1, 2) - 1.0;
    const torch::Tensor qq = q.unsqueeze(2) * q.unsqueeze(1) + exch;
    return {Gtt.mean(0), exch.mean(0), qq.mean(0)};
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
double ReducedObservables<Type, LatticeType>::holeRelation_(const typename Propagator::Spatial & P,
                                                           const typename Propagator::Spatial & H) const {
    // ppp_reduce.hole_relation_deviation: G_h(s, t)^T = -G_p(t, s) for t > s, and G_h(t, t)^T = 1 - G_p(t, t)
    double worst = 0;
    for (std::size_t i = 0; i < P.sources.size(); ++i) {
        const int64_t s = int64_t(P.sources[i]);
        if (s + 1 >= int64_t(Nt_)) { continue; }
        const torch::Tensor colP = torch::Tensor(NSL::Tensor<Type>(P.column[i], false)).slice(0, s + 1);
        const torch::Tensor rowH = torch::Tensor(NSL::Tensor<Type>(H.row[i], false)).slice(0, s + 1);
        const torch::Tensor b = -colP;
        worst = std::max(worst, (rowH.transpose(1, 2) - b).abs().max().template item<double>() / b.abs().max().template item<double>());
    }
    const torch::Tensor Gp = torch::Tensor(NSL::Tensor<Type>(P.diagonal, false));
    const torch::Tensor Gh = torch::Tensor(NSL::Tensor<Type>(H.diagonal, false));
    const torch::Tensor ref = torch::eye(int64_t(Nx_), cplx_) - Gp;
    return std::max(worst, (Gh.transpose(1, 2) - ref).abs().max().template item<double>() / ref.abs().max().template item<double>());
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void ReducedObservables<Type, LatticeType>::writeNonInteracting_(){
    // phi = 0, source time 0 only (as the NonInteracting block of the propagator program), double precision
    NSL::Tensor<Type> zero(Nt_, Nx_);
    zero = Type(0);
    const auto P = invM_.compute(zero, NSL::Hubbard::Particle, {0});
    const OneBody ob = oneBody_(P, {0});
    const torch::Tensor Fo = project_(ob.F), Bo = project_(ob.B);
    auto write = [&](const std::string & path, torch::Tensor t){
        t = t.contiguous();
        std::vector<std::size_t> dims(t.sizes().begin(), t.sizes().end());
        HighFive::DataSet ds = out_.createDataSet<std::complex<double>>(path, HighFive::DataSpace(dims));
        ds.write_raw(reinterpret_cast<const std::complex<double> *>(t.data_ptr<c10::complex<double>>()));
    };
    NSL::Stop::WriteGuard guard;
    for (const std::string & c: classNames_) {
        write("noninteracting/onebody/" + c + "/forward", block_(Fo, classIdx_.at(c), classIdx_.at(c)));
        write("noninteracting/onebody/" + c + "/backward", block_(Bo, classIdx_.at(c), classIdx_.at(c)));
    }
    const auto eq = equalTime_(ob.Gtt);
    write("noninteracting/equaltime/G", eq[0]);
    write("noninteracting/equaltime/ss", eq[1]);
    write("noninteracting/equaltime/qq", eq[2]);
    out_.flush();
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void ReducedObservables<Type, LatticeType>::stage2Config_(const typename Propagator::Spatial & P, const torch::Tensor & Gtt, double w){
    // ppp_reduce2: per source the window projections of G(s + t, s) and G(s, s + t), periodic (products of two
    // propagators), with the contact term G(s, s) - 1 at t = 0; connected exciton and pp matrices averaged over sources
    const int64_t Nt = int64_t(Nt_);
    const torch::Tensor t = torch::arange(Nt, long_);
    const double ns = double(P.sources.size());
    const torch::Tensor Iw = torch::eye(Uw_.size(1), cplx_);
    std::map<std::string, torch::Tensor> X, PP;
    for (std::size_t i = 0; i < P.sources.size(); ++i) {
        const torch::Tensor idx = (t + int64_t(P.sources[i])).remainder(Nt);
        const torch::Tensor Fo = projectWindow_(torch::Tensor(NSL::Tensor<Type>(P.column[i], false)).index_select(0, idx));
        torch::Tensor Bo = projectWindow_(torch::Tensor(NSL::Tensor<Type>(P.row[i], false)).index_select(0, idx));
        Bo[0] -= Iw;
        for (const auto & e: exc_) {
            // ppp_wick.exciton_pairs: X_ij = -Fo(c_i, c_j) Bo(v_j, v_i)
            const torch::Tensor x = -block_(Fo, e.c, e.c) * block_(Bo, e.v, e.v).transpose(1, 2);
            X[e.name] = (i == 0) ? x / ns : X[e.name] + x / ns;
        }
        for (const auto & p: pp_) {
            // ppp_wick.pp_pairs: P_ij = 2 al_i al_j [Fo(a_i,a_j) Fo(b_i,b_j) + sign Fo(a_i,b_j) Fo(b_i,a_j)]
            const torch::Tensor q = p.al2 * (block_(Fo, p.a, p.a) * block_(Fo, p.b, p.b)
                                             + (p.sign * block_(Fo, p.a, p.b)) * block_(Fo, p.b, p.a));
            PP[p.path] = (i == 0) ? q / ns : PP[p.path] + q / ns;
        }
    }
    // disconnected part from the equal-time density matrix Eo(t) = <c^+ c> = 1 - (Uw^T G(t,t) Uw)^T
    const torch::Tensor Eo = Iw - projectWindow_(Gtt).transpose(1, 2);
    const double Bd = double(B_);
    for (const auto & e: exc_) {
        const torch::Tensor o = torch::diagonal(block_(Eo, e.v, e.c), 0, 1, 2);     // <O_i(t)> = <c^+_v c_c>
        const torch::Tensor ob = torch::diagonal(block_(Eo, e.c, e.v), 0, 1, 2);    // <O^+_i(t)> = <c^+_c c_v>
        // D[t, i, j] = 1/Nt sum_t0 o[(t0 + t) % Nt, i] ob[t0, j] (ppp_reduce2.cross_correlation)
        const torch::Tensor D = torch::fft::ifft(torch::fft::fft(o, c10::nullopt, 0).unsqueeze(2)
                                                 * torch::fft::ifft(ob, c10::nullopt, 0).unsqueeze(1), c10::nullopt, 0);
        const torch::Tensor ex = torch::stack({o.mean(0), ob.mean(0)});
        accX_[e.name] += (w * X.at(e.name)) / Bd;
        accD_[e.name] += (w * D) / Bd;
        accEx_[e.name] += (w * ex) / Bd;
    }
    for (const auto & p: pp_) { accPP_[p.path] += (w * PP.at(p.path)) / Bd; }
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void ReducedObservables<Type, LatticeType>::checkContractions_(const typename Propagator::Spatial & P, const torch::Tensor & Gtt){
    // ppp_reduce2: the pair-basis contractions against the general ones (ppp_wick.exciton, ppp_wick.pp in the site
    // basis) at a few time separations from the first source, for one exciton and one pp pair
    const int64_t Nt = int64_t(Nt_), s0 = int64_t(P.sources.at(0));
    const torch::Tensor col = torch::Tensor(NSL::Tensor<Type>(P.column[0], false));
    const torch::Tensor row = torch::Tensor(NSL::Tensor<Type>(P.row[0], false));
    const torch::Tensor I = torch::eye(int64_t(Nx_), cplx_), Iw = torch::eye(Uw_.size(1), cplx_);
    auto E = [&](const torch::Tensor & G){ return I - G.transpose(0, 1); };                   // <c^+ c>
    const torch::Tensor Eo = Iw - projectWindow_(Gtt).transpose(1, 2);
    const torch::Tensor uc = U_.select(1, checkExc_[0]), uv = U_.select(1, checkExc_[1]);
    const torch::Tensor A = torch::outer(uc, uv);
    const int64_t pc = wpos_.at(checkExc_[0]), pv = wpos_.at(checkExc_[1]);
    const torch::Tensor ua = U_.select(1, checkPP_[0]), ub = U_.select(1, checkPP_[1]);
    const torch::Tensor Bm = (checkPP_[0] == checkPP_[1]) ? torch::outer(ua, ua)
        : (torch::outer(ua, ub) + torch::outer(ub, ua)) / std::sqrt(2.0);
    const int64_t pa = wpos_.at(checkPP_[0]), pb = wpos_.at(checkPP_[1]);
    const double al = (checkPP_[0] == checkPP_[1]) ? 0.5 : 1 / std::sqrt(2.0);
    auto c = [](const torch::Tensor & x){ return static_cast<std::complex<double>>(x.template item<c10::complex<double>>()); };
    for (const int64_t t: {int64_t(0), int64_t(1), Nt / 4, Nt / 2}) {
        const int64_t tg = (s0 + t) % Nt;
        const torch::Tensor Gf = col[tg];
        const torch::Tensor Gb = t ? row[tg] : row[tg] - I;                                     // contact term at t = 0
        const std::complex<double> Xg = -c(torch::einsum("xy,xa,ab,by->", {torch::conj(A), Gf, A, Gb}));
        const std::complex<double> Ot = c(torch::einsum("xy,yx->", {torch::conj(A), E(Gtt[tg])}));
        const std::complex<double> Od = c(torch::einsum("xy,xy->", {A, E(Gtt[s0])}));
        const std::complex<double> Sg = Xg + 2.0 * Ot * Od;
        const torch::Tensor Fo1 = torch::matmul(torch::matmul(UwT_, Gf), Uw_);
        const torch::Tensor Bo1 = torch::matmul(torch::matmul(UwT_, Gb), Uw_);
        const std::complex<double> Xf = -c(Fo1[pc][pc]) * c(Bo1[pv][pv]);
        const std::complex<double> Sf = Xf + 2.0 * c(Eo[tg][pv][pc]) * c(Eo[s0][pc][pv]);
        excDev_ = std::max({excDev_, std::abs(Xg - Xf) / std::abs(Xg), std::abs(Sg - Sf) / std::abs(Sg)});
        const std::complex<double> Pg = c(torch::einsum("xy,xa,ab,yb->", {torch::conj(Bm), Gf, Bm, Gf}));
        const std::complex<double> Pf = 2 * al * al * (c(Fo1[pa][pa]) * c(Fo1[pb][pb]) + c(Fo1[pa][pb]) * c(Fo1[pb][pa]));
        ppDev_ = std::max(ppDev_, std::abs(Pg - Pf) / std::abs(Pg));
    }
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void ReducedObservables<Type, LatticeType>::add(NSL::size_t cfgID, const NSL::Tensor<Type> & phi, Type S,
                                                double acceptanceProbability, double markovTime){
    if (cfgID != next()) {
        throw std::runtime_error(fmt::format("ReducedObservables: configuration {} given, {} expected next", cfgID, next()));
    }
    const int64_t Nt = int64_t(Nt_);

    // propagators of the particle species (the hole species is redundant at mu = 0 on a bipartite lattice)
    const auto P = invM_.compute(phi, NSL::Hubbard::Particle, sources_);
    const OneBody ob = oneBody_(P, sources_);
    const torch::Tensor Fo = project_(ob.F), Bo = project_(ob.B);
    const auto eq = equalTime_(ob.Gtt);

    // field: gauge action and field modes (ppp_autocorr conventions)
    const torch::Tensor pr = torch::real(torch::Tensor(NSL::Tensor<Type>(phi, false))).to(torch::kDouble);
    const double Sg = 0.5 * (torch::matmul(pr, Ainv_) * pr).sum().template item<double>();
    const torch::Tensor modes = torch::matmul(pr.sum(0), QV_) / torch::sqrt(double(Nt) * delta_ * lamV_);

    // determinant sign and fermion action at phi (ppp_reduce: logdet_particle)
    signed char sign = 0;
    double residual = 0, SF = std::numeric_limits<double>::quiet_NaN();
    if (doSign_) {
        invM_.hfm_.populate(phi, NSL::Hubbard::Particle);
        const std::complex<double> logdet = static_cast<std::complex<double>>(invM_.hfm_.logDetM());
        const std::complex<double> r = std::exp(std::complex<double>(0, logdet.imag() - 0.5 * pr.sum().template item<double>()));
        if (std::isfinite(logdet.real()) && std::isfinite(logdet.imag()) && std::isfinite(r.real()) && std::isfinite(r.imag())) {
            sign = r.real() > 0 ? 1 : -1;
            residual = std::abs(r.imag());
            SF = -2.0 * logdet.real();
            signRes_ = std::max(signRes_, residual);
            if (!bridge_) {
                sfDev_ = std::max(sfDev_, std::abs(static_cast<std::complex<double>>(S).real() - Sg - SF));
            }
        } else {
            residual = std::numeric_limits<double>::quiet_NaN();
            signRes_ = std::numeric_limits<double>::quiet_NaN();
        }
    }

    // bridge link: Lambda from the generalized density at the bridge slice, or averaged over all slices for the
    // multi-slice bridge (ppp_reduce.bridge_lambda)
    double w = 1.0, lam = std::numeric_limits<double>::quiet_NaN();
    if (bridge_) {
        const torch::Tensor n = slice_ == allSlices ? 1.0 - torch::diagonal(ob.Gtt, 0, -2, -1)        // (Nt, Nx)
                                                    : 1.0 - torch::diagonal(ob.Gtt[int64_t(slice_)]);  // (Nx)
        const std::complex<double> e(std::cos(theta_), std::sin(theta_));
        const torch::Tensor a = n.mul(c10::Scalar(c10::complex<double>(e.real() - 1, e.imag()))).add(1.0).abs().pow(2);
        const torch::Tensor b = n.mul(c10::Scalar(c10::complex<double>(e.real() - 1, -e.imag()))).add(1.0).abs().pow(2);
        lam = 0.5 * (a + b).mean().template item<double>();
        w = 1.0 / lam;
        const double dev = ((torch::real(n) - 0.5).abs() / torch::clamp_min(n.abs(), 1.0)).max().template item<double>();
        reNDev_ = std::max(reNDev_, dev);
    }

    // checks on the first configurations: particle-hole relation, pair contractions
    if (cfgID < checkFirst_) {
        const auto H = invM_.compute(phi, NSL::Hubbard::Hole, sources_);
        holeDev_ = std::max(holeDev_, holeRelation_(P, H));
        holeConfigs_ = std::max<int64_t>(holeConfigs_, int64_t(cfgID) + 1);
        if (out2_) {
            checkContractions_(P, ob.Gtt);
            checkedConfigs_ = std::max<int64_t>(checkedConfigs_, int64_t(cfgID) + 1);
        }
    }

    // accumulate the bin
    if (pending_ == 0) {
        binStart_ = cfgID;
        for (const std::string & c: classNames_) {
            const int64_t m = classIdx_.at(c).size(0);
            accF_[c] = torch::zeros({Nt, m, m}, cplx_);
            accB_[c] = torch::zeros({Nt, m, m}, cplx_);
        }
        for (auto & a: accEq_) { a = torch::zeros({int64_t(Nx_), int64_t(Nx_)}, cplx_); }
        for (const auto & e: exc_) {
            const int64_t n = e.c.size(0);
            accX_[e.name] = torch::zeros({Nt, n, n}, cplx_);
            accD_[e.name] = torch::zeros({Nt, n, n}, cplx_);
            accEx_[e.name] = torch::zeros({2, n}, cplx_);
        }
        for (const auto & p: pp_) { accPP_[p.path] = torch::zeros({Nt, p.a.size(0), p.a.size(0)}, cplx_); }
        wsum_ = 0;
    }
    for (const std::string & c: classNames_) {
        const torch::Tensor & idx = classIdx_.at(c);
        const torch::Tensor Fc = block_(Fo, idx, idx), Bc = block_(Bo, idx, idx);
        accF_[c] += w * Fc;
        accB_[c] += w * Bc;
        // addition probe A_aa(t) = [F_aa(t) - B_aa(Nt - t)] / 2 (t > 0), F_aa(0) at t = 0; not weighted
        const torch::Tensor dF = torch::real(torch::diagonal(Fc.index_select(0, probe_), 0, 1, 2));
        const torch::Tensor dB = torch::real(torch::diagonal(Bc.index_select(0, back_), 0, 1, 2));
        probes_[c].push_back(torch::where(probePositive_, 0.5 * (dF - dB), dF));
    }
    for (int k = 0; k < 3; ++k) { accEq_[k] += w * eq[k]; }
    if (out2_) { stage2Config_(P, ob.Gtt, w); }
    wsum_ += w;

    ids_.push_back(int64_t(cfgID));
    S_.push_back(static_cast<std::complex<double>>(S));
    P_.push_back(acceptanceProbability);
    T_.push_back(markovTime);
    Sg_.push_back(Sg);
    modes_.push_back(modes);
    sign_.push_back(sign);
    res_.push_back(residual);
    SF_.push_back(SF);
    lam_.push_back(lam);
    ++pending_;
    if (pending_ == B_) { writeBin_(); }
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
template<typename T>
void ReducedObservables<Type, LatticeType>::append_(HighFive::File & f, const std::string & path, const T * data, NSL::size_t rows, NSL::size_t at){
    HighFive::DataSet ds = f.getDataSet(path);
    std::vector<std::size_t> dims = ds.getDimensions();
    dims[0] = at + rows;
    ds.resize(dims);
    std::vector<std::size_t> offset(dims.size(), 0), count = dims;
    offset[0] = at;
    count[0] = rows;
    ds.select(offset, count).write_raw(data);
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void ReducedObservables<Type, LatticeType>::appendComplex_(HighFive::File & f, bool single, const std::string & path,
                                                           const torch::Tensor & t, NSL::size_t rows, NSL::size_t at){
    if (single) {
        const torch::Tensor c = t.to(torch::kComplexFloat).contiguous();
        append_(f, path, reinterpret_cast<const std::complex<float> *>(c.data_ptr<c10::complex<float>>()), rows, at);
    } else {
        const torch::Tensor c = t.to(torch::kComplexDouble).contiguous();
        append_(f, path, reinterpret_cast<const std::complex<double> *>(c.data_ptr<c10::complex<double>>()), rows, at);
    }
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void ReducedObservables<Type, LatticeType>::appendReal_(HighFive::File & f, bool single, const std::string & path,
                                                        const torch::Tensor & t, NSL::size_t rows, NSL::size_t at){
    if (single) {
        const torch::Tensor c = t.to(torch::kFloat).contiguous();
        append_(f, path, c.data_ptr<float>(), rows, at);
    } else {
        const torch::Tensor c = t.to(torch::kDouble).contiguous();
        append_(f, path, c.data_ptr<double>(), rows, at);
    }
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void ReducedObservables<Type, LatticeType>::writeBin_(){
    NSL::Stop::WriteGuard guard;      // a SIGTERM waits until the bin is complete
    const NSL::size_t b = binStart_ / B_;
    if (nBins_ == b) { writeStage1_(); }
    if (out2_ && nBins2_ == b) { writeStage2_(); }
    pending_ = 0;
    ids_.clear(); S_.clear(); P_.clear(); T_.clear(); Sg_.clear(); modes_.clear(); sign_.clear(); res_.clear();
    SF_.clear(); lam_.clear();
    for (auto & [c, v]: probes_) { v.clear(); }
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void ReducedObservables<Type, LatticeType>::writeStage1_(){
    const NSL::size_t c0 = nConfigs_, b0 = nBins_;
    const double Bd = double(B_);
    HighFive::File & f = out_;

    // per configuration
    append_(f, "config/id", ids_.data(), B_, c0);
    append_(f, "config/S", S_.data(), B_, c0);
    append_(f, "config/acceptanceProbability", P_.data(), B_, c0);
    append_(f, "config/markovTime", T_.data(), B_, c0);
    append_(f, "config/S_gauge", Sg_.data(), B_, c0);
    const torch::Tensor modes = torch::stack(modes_).contiguous();
    append_(f, "config/field_modes", modes.data_ptr<double>(), B_, c0);
    append_(f, "config/detsign", sign_.data(), B_, c0);
    append_(f, "config/detsign_residual", res_.data(), B_, c0);
    if (doSign_) { append_(f, "config/S_fermion_phi", SF_.data(), B_, c0); }
    if (bridge_) { append_(f, "config/bridge_lambda", lam_.data(), B_, c0); }

    // per bin: averages of w O over the bin, and of w
    const double wbin = wsum_ / Bd;
    append_(f, "weights", &wbin, 1, b0);
    for (const std::string & c: classNames_) {
        appendComplex_(f, single_, "onebody/" + c + "/forward", accF_.at(c) / Bd, 1, b0);
        appendComplex_(f, single_, "onebody/" + c + "/backward", accB_.at(c) / Bd, 1, b0);
        appendReal_(f, single_, "onebody/" + c + "/addition_probe", torch::stack(probes_.at(c)), B_, c0);
    }
    appendComplex_(f, single_, "equaltime/G", accEq_[0] / Bd, 1, b0);
    appendComplex_(f, single_, "equaltime/ss", accEq_[1] / Bd, 1, b0);
    appendComplex_(f, single_, "equaltime/qq", accEq_[2] / Bd, 1, b0);

    // then the counters and checks
    nConfigs_ += B_;
    nBins_ += 1;
    f.getAttribute("n_configs").write(int64_t(nConfigs_));
    f.getAttribute("n_bins").write(int64_t(nBins_));
    HighFive::Group checks = f.getGroup("checks");
    checks.getAttribute("hole_relation_max_dev").write(holeDev_);
    checks.getAttribute("hole_relation_configs").write(holeConfigs_);
    if (doSign_) { checks.getAttribute("detsign_max_residual").write(signRes_); }
    if (bridge_) { checks.getAttribute("bridge_re_n_max_dev").write(reNDev_); }
    if (doSign_ && !bridge_) { checks.getAttribute("S_fermion_phi_max_dev").write(sfDev_); }
    f.flush();
}

template<NSL::Concept::isNumber Type, NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType>
void ReducedObservables<Type, LatticeType>::writeStage2_(){
    const NSL::size_t b0 = nBins2_;
    HighFive::File & f = *out2_;
    const int64_t first = int64_t(binStart_);
    append_(f, "config/first_id", &first, 1, b0);
    const double wbin = wsum_ / double(B_);
    append_(f, "weights", &wbin, 1, b0);
    for (const auto & e: exc_) {
        appendComplex_(f, single2_, "exciton/" + e.name + "/connected", accX_.at(e.name), 1, b0);
        appendComplex_(f, single2_, "exciton/" + e.name + "/disconnected", accD_.at(e.name), 1, b0);
        appendComplex_(f, false, "exciton/" + e.name + "/expectation", accEx_.at(e.name), 1, b0);
    }
    for (const auto & p: pp_) { appendComplex_(f, single2_, p.path + "/correlator", accPP_.at(p.path), 1, b0); }

    nConfigs2_ += B_;
    nBins2_ += 1;
    f.getAttribute("n_configs").write(int64_t(nConfigs2_));
    f.getAttribute("n_bins").write(int64_t(nBins2_));
    HighFive::Group checks = f.getGroup("checks");
    checks.getAttribute("exciton_max_rel_dev").write(excDev_);
    checks.getAttribute("pp_max_rel_dev").write(ppDev_);
    checks.getAttribute("checked_configs").write(checkedConfigs_);
    f.flush();
}

} // namespace NSL::Measure::Hubbard

#endif // NSL_REDUCED_OBSERVABLES_TPP
