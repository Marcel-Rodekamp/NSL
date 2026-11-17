#ifndef NSL_MARKOVCHAIN_AUTOTUNE_TPP
#define NSL_MARKOVCHAIN_AUTOTUNE_TPP

/*! \file autotune.tpp
 *  \brief Thermalization, choice of the number of MD steps (Nmd), and production of a Markov chain.
 *
 *  NSL::MCMC::Autotune::run drives a RadialHMC through three phases. All of them can be stopped
 *  (NSL::Stop) and continued by running the same command again; the checkpoint is kept in
 *  <base>/thermalization until the production has started.
 *
 *  1. Thermalization: Ntherm combined updates (radial + HMC). When Nmd is tuned, Nmd doubles after
 *     `escalate after` rejections with dH > `spike threshold` in a row, so that a start far from
 *     equilibrium does not get stuck. With a fixed Nmd (plain run) only a warning is given.
 *  2. Tuning (only if Nmd is tuned), on equilibrium data. A trajectory with dH > `spike threshold`
 *     is a "spike"; the others form the "bulk". For a second-order integrator the bulk acceptance
 *     follows P_b(Nmd) = erfc((N0/Nmd)^2/2) with one constant N0, the "step scale" (found constant
 *     over Nmd for benzene, perylene and a 140-site zigzag ribbon). Each batch of `batch`
 *     trajectories gives N0 = Nmd sqrt(2 erfcinv(P_b)); pooled over all Nmd tried, it proposes
 *     Nmd* = N0/sqrt(2 erfcinv(target)), at most a factor 2 up or 4 down per batch, and doubles Nmd
 *     while P_b < 0.05. The search ends when Nmd* equals Nmd or lies within tolerance*Nmd of it, and
 *     Nmd has `min trajectories` bulk trajectories (if Nmd* alternates between two Nmd that both have
 *     them, the latest Nmd* is taken). If more than `spike fraction` of the trajectories at
 *     that Nmd are spikes, Nmd and 1.5 Nmd are run alternately on the same chain (`min
 *     trajectories` each) and the one with the lower cost per independent configuration,
 *     (Nmd + 1)(2/P - 1) MD steps with P the mean acceptance, is kept; this is repeated upward at
 *     most 3 times. (Steps, not seconds, so that a seeded run makes the same choice.)
 *  3. Production: Nconf configurations with the final Nmd in <base>/markovChain, followed by
 *     diagnostics of the stored trajectories (acceptance, spike fraction, bulk acceptance,
 *     <e^{-dH}>, which is 1 for a correct integrator), also written to <base>/Meta/diagnostics.
 *
 *  Every trajectory of phases 1 and 2 is recorded in <base>/Meta/autotune/trajectories (nmd,
 *  expMinusDH, accepted, seconds, stage: 0 thermalization, 1 search, 2.. comparison rounds); the
 *  result is in <base>/Meta/autotune/result, and <base>/Meta/params/nMD holds the Nmd used.
 *
 *  yml interface (all optional, defaults in brackets):
 *    HMC:      autotune: true | false | legacy [true],
 *              start: cold | gaussian [cold, except gaussian for a plain run with the leapfrog]
 *    Leapfrog: Nmd [10 when tuned; required otherwise]
 *    Autotune: target bulk acceptance [0.85], spike threshold [3], batch [20],
 *              min trajectories [50], tolerance [0.1], max trajectories [2000],
 *              spike fraction [0.1], escalate after [3], max Nmd [4096], checkpoint every [10]
 *  `autotune: legacy`, or the environment variable NSL_AUTOTUNE_PROTOCOL=legacy (set by
 *  autotuner8/autotuner.py), selects the protocol of autotuner8/autotuner.py instead.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "IO.hpp"
#include "logger.hpp"
#include "stop.hpp"
#include "MarkovChain/markovState.tpp"

namespace NSL::MCMC::Autotune {

inline constexpr double pi = 3.14159265358979323846;

//! Inverse of the complementary error function for 0 < x < 2 (safeguarded Newton iteration).
inline double erfcinv(double x){
    if (!(x > 0. && x < 2.)) {
        throw std::domain_error("NSL::MCMC::Autotune::erfcinv: the argument must lie in (0, 2)");
    }
    double lo = -10., hi = 10., z = 0.;
    for (int i = 0; i < 200; ++i) {
        const double f = std::erfc(z) - x;      // erfc decreases: f > 0 means z is too small
        if (f == 0.) { break; }
        if (f > 0.) { lo = z; } else { hi = z; }
        const double dfdz = -2. / std::sqrt(pi) * std::exp(-z * z);
        double next = z - f / dfdz;
        if (!(next > lo && next < hi)) { next = 0.5 * (lo + hi); }
        if (std::abs(next - z) <= 1e-15 * (1. + std::abs(z))) { z = next; break; }
        z = next;
    }
    return z;
}

//! What a run does with Nmd: tune it (default), keep the yml value (plain run), or follow the
//! protocol of autotuner8/autotuner.py (legacy).
enum class Mode { Tune, Plain, Legacy };

inline Mode modeFromYAML(const YAML::Node & yml){
    if (const char * env = std::getenv("NSL_AUTOTUNE_PROTOCOL"); env && std::string(env) == "legacy") {
        return Mode::Legacy;
    }
    if (!yml["HMC"] || !yml["HMC"]["autotune"]) { return Mode::Tune; }
    const YAML::Node a = yml["HMC"]["autotune"];
    if (a.as<std::string>() == "legacy") { return Mode::Legacy; }
    bool tune = true;
    try {
        tune = a.as<bool>();
    } catch (const YAML::Exception &) {
        throw std::invalid_argument("`HMC: autotune:` must be true, false or legacy, not '" + a.as<std::string>() + "'");
    }
    return tune ? Mode::Tune : Mode::Plain;
}

inline std::string to_string(Mode m){
    switch (m) {
        case Mode::Tune:   return "tune Nmd";
        case Mode::Plain:  return "plain run (fixed Nmd)";
        case Mode::Legacy: return "legacy protocol of autotuner8/autotuner.py";
    }
    return "";
}

//! Starting field: phi = 0 ("cold") or a sample of the gauge action ("gaussian").
//! The default depends on the integrator (see startDefault).
enum class Start { Cold, Gaussian };

/*! Default start: cold, except for a plain run (fixed Nmd) with the leapfrog.
 *  A gaussian start lies in a rough, slowly relaxing region: a 140-site ribbon needed Nmd >= 40
 *  and chains stayed stuck, and gaussian-started chains showed more spikes and larger errors for
 *  a long time (perylene, Hubbard benzene), while cold starts thermalized in 1-2 trajectories.
 *  With the leapfrog, however, every Gaussian mode starts at zero displacement from phi = 0, so the
 *  leapfrog error of the quadratic part of the action has the same sign in every trajectory and
 *  grows like V eps^2 (dH = 4.2 for perylene at Nmd 10, 11 for the ribbon at Nmd 20: no trajectory
 *  accepted). When Nmd is tuned, one doubling of Nmd during the thermalization removes this
 *  (perylene: 10 -> 20); with a fixed Nmd the chain would stay at phi = 0, hence gaussian there.
 *  Fourier acceleration integrates the quadratic part exactly.
 */
inline Start startDefault(bool fourierAcceleration, bool tuneNmd){
    return (fourierAcceleration || tuneNmd) ? Start::Cold : Start::Gaussian;
}

inline Start startFromYAML(const YAML::Node & yml, Start fallback){
    if (!yml["HMC"] || !yml["HMC"]["start"]) { return fallback; }
    const std::string s = yml["HMC"]["start"].as<std::string>();
    if (s == "cold") { return Start::Cold; }
    if (s == "gaussian") { return Start::Gaussian; }
    throw std::invalid_argument("`HMC: start:` must be cold or gaussian, not '" + s + "'");
}

//! Settings of the tuner (yml block `Autotune`, see the file documentation).
struct Options {
    double target = 0.85;
    double spikeThreshold = 3.;
    NSL::size_t batch = 20;
    NSL::size_t minTrajectories = 50;
    double tolerance = 0.1;
    NSL::size_t maxTrajectories = 2000;
    double spikeFraction = 0.1;
    NSL::size_t escalateAfter = 3;
    NSL::size_t maxNmd = 4096;
    NSL::size_t checkpointEvery = 10;

    static Options fromYAML(const YAML::Node & yml){
        Options o;
        const YAML::Node a = yml["Autotune"];
        if (!a) { return o; }
        auto get = [&](const char * key, auto & value){
            if (a[key]) { value = a[key].as<std::decay_t<decltype(value)>>(); }
        };
        get("target bulk acceptance", o.target);
        get("spike threshold", o.spikeThreshold);
        get("batch", o.batch);
        get("min trajectories", o.minTrajectories);
        get("tolerance", o.tolerance);
        get("max trajectories", o.maxTrajectories);
        get("spike fraction", o.spikeFraction);
        get("escalate after", o.escalateAfter);
        get("max Nmd", o.maxNmd);
        get("checkpoint every", o.checkpointEvery);
        if (!(o.target > 0.05 && o.target < 0.999)) { throw std::invalid_argument("`Autotune: target bulk acceptance` must lie in (0.05, 0.999)"); }
        if (!(o.spikeThreshold > 0)) { throw std::invalid_argument("`Autotune: spike threshold` must be positive"); }
        if (o.batch < 1 || o.minTrajectories < 1 || o.escalateAfter < 1 || o.maxNmd < 1 || o.checkpointEvery < 1) {
            throw std::invalid_argument("`Autotune:` batch, min trajectories, escalate after, max Nmd and checkpoint every must be positive");
        }
        if (o.tolerance < 0 || o.spikeFraction < 0 || o.spikeFraction > 1) {
            throw std::invalid_argument("`Autotune:` tolerance must be >= 0 and spike fraction in [0, 1]");
        }
        return o;
    }
};

//! Parameters of the Markov chain.
struct RunParameters {
    NSL::size_t Nmd;            //!< fixed Nmd (plain run) or starting value (tuning)
    NSL::size_t Ntherm;         //!< thermalization updates
    NSL::size_t Nconf;          //!< configurations of the production
    NSL::size_t saveFrequency;
    NSL::size_t Nradial;
    NSL::size_t Nhmc;
    double radialScale;
};

//! One entry per trajectory of thermalization and tuning.
struct Record {
    std::vector<double> nmd, p, accepted, seconds, stage;   // p = e^{-dH} as stored by the HMC

    void add(NSL::size_t n, double pp, bool acc, double secs, int st){
        nmd.push_back(double(n)); p.push_back(pp); accepted.push_back(acc ? 1. : 0.);
        seconds.push_back(secs); stage.push_back(double(st));
    }
    std::size_t size() const { return p.size(); }

    void write(NSL::H5IO & h5, const std::string & node) const {
        if (p.empty()) { return; }
        NSL::Stop::WriteGuard writing;
        HighFive::File & file = h5.getFile();
        for (auto [name, v] : {std::pair<const char *, const std::vector<double> *>{"nmd", &nmd}, {"expMinusDH", &p},
                               {"accepted", &accepted}, {"seconds", &seconds}, {"stage", &stage}}) {
            const std::string full = node + "/" + name;
            if (file.exist(full)) { file.unlink(full); }
            file.createDataSet<double>(full, HighFive::DataSpace::From(*v)).write(*v);
        }
        file.flush();
    }
    void read(NSL::H5IO & h5, const std::string & node){
        HighFive::File & file = h5.getFile();
        if (!file.exist(node + "/expMinusDH")) { return; }
        file.getDataSet(node + "/nmd").read(nmd);
        file.getDataSet(node + "/expMinusDH").read(p);
        file.getDataSet(node + "/accepted").read(accepted);
        file.getDataSet(node + "/seconds").read(seconds);
        file.getDataSet(node + "/stage").read(stage);
    }
};

//! Statistics of the trajectories at one Nmd.
struct Stats {
    NSL::size_t n = 0, nBulk = 0;
    double sumPb = 0, sumPb2 = 0, sumPall = 0, seconds = 0;

    void add(double p, double secs, double spikeThreshold){
        const double a = std::min(1., p);
        ++n; sumPall += a; seconds += secs;
        if (p >= std::exp(-spikeThreshold)) { ++nBulk; sumPb += a; sumPb2 += a * a; }
    }
    double spikeFraction() const { return n ? 1. - double(nBulk) / double(n) : 0.; }
    double bulkAcceptance() const { return nBulk ? sumPb / double(nBulk) : 0.; }
    double bulkAcceptanceError() const {
        if (nBulk < 2) { return 1.; }
        const double var = (sumPb2 - sumPb * sumPb / double(nBulk)) / double(nBulk - 1);
        return std::sqrt(std::max(var, 1e-4) / double(nBulk));   // standard deviation at least 0.01
    }
    double acceptance() const { return n ? sumPall / double(n) : 0.; }
    double secondsPerTrajectory() const { return n ? seconds / double(n) : 0.; }
    //! cost per independent configuration at Nmd = nmd in MD steps, (Nmd + 1) (2/P - 1), if every
    //! accepted trajectory decorrelates; the +1 counts the action evaluation of each trajectory.
    //! Counted in steps rather than seconds, so that a seeded run always makes the same choice.
    double cost(NSL::size_t nmd) const {
        const double P = acceptance();
        return P > 0 ? double(nmd + 1) * (2. / P - 1.) : std::numeric_limits<double>::infinity();
    }
};

//! The step scale N0 from the bulk acceptance at one Nmd, with its error (NaN if not usable).
inline std::pair<double, double> stepScale(NSL::size_t nmd, const Stats & s){
    if (s.nBulk < 5 || s.bulkAcceptance() < 0.05) { return {NAN, NAN}; }
    const double pb = std::min(s.bulkAcceptance(), 1. - 0.5 / double(s.nBulk));
    const double e = erfcinv(pb);
    const double n0 = double(nmd) * std::sqrt(2. * e);
    // dN0/dP_b = Nmd/sqrt(2e) * derfcinv/dP_b, derfcinv/dx = -sqrt(pi)/2 exp(e^2)
    const double dn0 = double(nmd) / std::sqrt(2. * e) * std::sqrt(pi) / 2. * std::exp(e * e) * s.bulkAcceptanceError();
    return {n0, dn0};
}

//! Error-weighted mean of the step scale over all Nmd with usable statistics.
inline std::pair<double, double> pooledStepScale(const std::map<NSL::size_t, Stats> & stats){
    double sw = 0, swx = 0;
    for (const auto & [nmd, s] : stats) {
        const auto [n0, dn0] = stepScale(nmd, s);
        if (std::isfinite(n0) && dn0 > 0) { sw += 1. / (dn0 * dn0); swx += n0 / (dn0 * dn0); }
    }
    if (sw == 0) { return {NAN, NAN}; }
    return {swx / sw, 1. / std::sqrt(sw)};
}

//! Checkpoint of thermalization and tuning (scalars in <base>/thermalization).
struct Checkpoint {
    NSL::size_t phase = 0;           // 0 thermalization, 1 tuning, 2 done
    NSL::size_t completedSteps = 0;  // thermalization updates done
    NSL::size_t nmd = 1;             // current Nmd
    NSL::size_t compare = 0;         // 0: search; k > 0: comparison round k of low and high
    NSL::size_t low = 0, high = 0;
};

//! Diagnostics of the stored trajectories of <base>/markovChain (logged and written to
//! <base>/Meta/diagnostics).
inline void diagnose(NSL::H5IO & h5, const std::string & base, const Options & opt){
    const std::string chain = base + "/markovChain";
    if (!h5.exist(chain)) { return; }
    auto [minCfg, maxCfg] = h5.getMinMaxConfigs(chain);
    HighFive::File & file = h5.getFile();
    Stats s;
    double sumP = 0, sumP2 = 0;
    for (NSL::size_t k = minCfg + 1; k <= maxCfg; ++k) {   // configuration minCfg is the start of the chain
        const std::string node = fmt::format("{}/{}/acceptanceProbability", chain, k);
        if (!file.exist(node)) { continue; }
        double p = 0;
        file.getDataSet(node).read(p);
        if (!std::isfinite(p)) { p = 0; }
        s.add(p, 0., opt.spikeThreshold);
        sumP += p; sumP2 += p * p;
    }
    if (s.n == 0) { return; }
    const double meanP = sumP / double(s.n);
    const double errP = s.n > 1 ? std::sqrt(std::max(0., sumP2 / double(s.n) - meanP * meanP) / double(s.n - 1)) : NAN;
    NSL::Logger::info("Diagnostics of {} ({} stored trajectories): acceptance {:.3f}, spike fraction {:.3f} (dH > {}), "
                      "bulk acceptance {:.3f}, <e^-dH> = {:.3f} +- {:.3f} (1 for a correct integrator)",
                      chain, s.n, s.acceptance(), s.spikeFraction(), opt.spikeThreshold, s.bulkAcceptance(), meanP, errP);
    if (s.spikeFraction() > opt.spikeFraction) {
        NSL::Logger::warn("{:.0f}% of the trajectories are spikes (dH > {}); a larger Nmd may be cheaper per independent configuration.",
                          100. * s.spikeFraction(), opt.spikeThreshold);
    }
    if (s.n >= 100 && std::abs(meanP - 1.) > 3. * errP) {
        NSL::Logger::warn("<e^-dH> = {:.3f} +- {:.3f} differs from 1 by more than 3 standard deviations: check the integrator "
                          "(or rare large negative dH are missing from the sample).", meanP, errP);
    }
    NSL::Logger::info("Independent chains (another seed or file) are the check for slowly decorrelating regions: "
                      "compare <S> and the spike fraction between chains.");
    NSL::Stop::WriteGuard writing;
    const std::string d = base + "/Meta/diagnostics";
    h5.deleteData(d);
    h5.write(double(s.n), d + "/trajectories");
    h5.write(s.acceptance(), d + "/acceptance");
    h5.write(s.spikeFraction(), d + "/spikeFraction");
    h5.write(opt.spikeThreshold, d + "/spikeThreshold");
    h5.write(s.bulkAcceptance(), d + "/bulkAcceptance");
    h5.write(meanP, d + "/expMinusDH");
    h5.write(std::isfinite(errP) ? errP : 0., d + "/expMinusDHError");
}

namespace detail {

template<typename HMCType, NSL::Concept::isNumber Type>
struct ChainRun {
    HMCType & hmc;
    NSL::H5IO & h5;
    std::string base;
    RunParameters rp;
    Options opt;
    Record record;
    Checkpoint cp;
    NSL::MCMC::MarkovState<Type> state;
    NSL::size_t sinceCheckpoint = 0;

    std::string checkpointNode() const { return base + "/thermalization"; }
    std::string recordNode() const { return base + "/Meta/autotune"; }

    //! one combined update with Nmd = n, recorded with the given stage
    void trajectory(NSL::size_t n, int stage){
        hmc.setNumberSteps(n);
        std::vector<double> ps;
        const auto t0 = std::chrono::steady_clock::now();
        state = hmc.step(state, rp.Nradial, rp.Nhmc, NSL::RealTypeOf<Type>(rp.radialScale), &ps);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        record.add(n, ps.empty() ? 1. : ps.back(), state.accepted, secs, stage);
        ++sinceCheckpoint;
    }

    void writeCheckpoint(){
        NSL::Stop::WriteGuard writing;   // a SIGTERM waits until the checkpoint is complete
        const std::string c = checkpointNode();
        h5.deleteData(c);
        h5.write(state, c + "/state/0");
        h5.write(cp.completedSteps, c + "/completedSteps");
        h5.write(cp.phase, c + "/phase");
        h5.write(cp.nmd, c + "/nmd");
        h5.write(cp.compare, c + "/compare");
        h5.write(cp.low, c + "/low");
        h5.write(cp.high, c + "/high");
        record.write(h5, recordNode() + "/trajectories");
        sinceCheckpoint = 0;
    }

    //! true if a stop was requested (checkpoint written); also writes periodic checkpoints
    bool stopRequested(){
        if (NSL::Stop::requested()) {
            writeCheckpoint();
            NSL::Logger::warn("Stopped ({}) in the {}; stored in {}. Run the same command again to continue.",
                NSL::Stop::signalName(), cp.phase == 0 ? "thermalization" : "tuning", checkpointNode());
            return true;
        }
        if (sinceCheckpoint >= opt.checkpointEvery) { writeCheckpoint(); }
        return false;
    }

    std::map<NSL::size_t, Stats> stats(int stage) const {
        std::map<NSL::size_t, Stats> s;
        for (std::size_t i = 0; i < record.size(); ++i) {
            if (int(record.stage[i]) == stage) {
                s[NSL::size_t(record.nmd[i])].add(record.p[i], record.seconds[i], opt.spikeThreshold);
            }
        }
        return s;
    }

    NSL::size_t tuningTrajectories() const {
        return NSL::size_t(std::count_if(record.stage.begin(), record.stage.end(), [](double st){ return st >= 1; }));
    }

    //! Thermalization; false if stopped on a signal
    bool thermalize(bool escalate){
        NSL::size_t largeInARow = 0;
        bool warned = false;
        const NSL::size_t logEvery = std::max<NSL::size_t>(1, rp.Ntherm / 20);
        if (cp.completedSteps < rp.Ntherm) {
            NSL::Logger::info("Thermalization: {} of {} updates done, Nmd = {}", cp.completedSteps, rp.Ntherm, cp.nmd);
        }
        while (cp.completedSteps < rp.Ntherm) {
            trajectory(cp.nmd, 0);
            ++cp.completedSteps;
            const bool large = !state.accepted && !(record.p.back() >= std::exp(-opt.spikeThreshold));
            largeInARow = large ? largeInARow + 1 : 0;
            if (largeInARow >= opt.escalateAfter) {
                if (escalate) {
                    if (cp.nmd >= opt.maxNmd) {
                        throw std::runtime_error(fmt::format("Thermalization: rejected with dH > {} even at Nmd = {} (max Nmd)",
                                                             opt.spikeThreshold, cp.nmd));
                    }
                    cp.nmd = std::min(2 * cp.nmd, opt.maxNmd);
                    NSL::Logger::warn("Thermalization: {} rejections with dH > {} in a row; Nmd -> {}",
                                      largeInARow, opt.spikeThreshold, cp.nmd);
                } else if (!warned) {
                    NSL::Logger::warn("Thermalization: {} rejections with dH > {} in a row at the fixed Nmd = {}; "
                                      "consider `HMC: start: cold` or a larger Nmd.", largeInARow, opt.spikeThreshold, cp.nmd);
                    warned = true;
                }
                largeInARow = 0;
            }
            if (cp.completedSteps % logEvery == 0 || cp.completedSteps == rp.Ntherm) {
                NSL::Logger::info("Thermalization {}/{} (Nmd {}): S = {:.6g}, e^-dH = {:.3g}, accepted {}",
                    cp.completedSteps, rp.Ntherm, cp.nmd, NSL::real(state.actionValue), record.p.back(), state.accepted);
            }
            if (stopRequested()) { return false; }
        }
        return true;
    }

    NSL::size_t higher(NSL::size_t n) const {
        return std::min(opt.maxNmd, std::max(n + 1, NSL::size_t(std::llround(1.5 * double(n)))));
    }

    //! Tuning of Nmd; false if stopped on a signal. On return cp.nmd is the result.
    bool tune(){
        const double eStar = std::sqrt(2. * erfcinv(opt.target));
        NSL::Logger::info("Tuning Nmd: target bulk acceptance {}, spike threshold dH > {}, batches of {}",
                          opt.target, opt.spikeThreshold, opt.batch);
        while (true) {
            if (tuningTrajectories() >= opt.maxTrajectories) {
                if (cp.compare > 0) {
                    // keep the cheaper of the two candidates measured so far
                    const auto S = stats(1 + int(cp.compare));
                    const double cl = S.count(cp.low) ? S.at(cp.low).cost(cp.low) : std::numeric_limits<double>::infinity();
                    const double ch = S.count(cp.high) ? S.at(cp.high).cost(cp.high) : std::numeric_limits<double>::infinity();
                    cp.nmd = (ch < cl) ? cp.high : cp.low;
                }
                NSL::Logger::warn("Tuning: the budget of {} trajectories is used up; keeping Nmd = {}. "
                                  "Check <name>/Meta/autotune and consider a larger `Autotune: max trajectories`.",
                                  opt.maxTrajectories, cp.nmd);
                break;
            }
            if (cp.compare == 0) {
                // search with the bulk model
                for (NSL::size_t k = 0; k < opt.batch; ++k) {
                    trajectory(cp.nmd, 1);
                    if (stopRequested()) { return false; }
                }
                const auto S = stats(1);
                const Stats & s = S.at(cp.nmd);
                const auto [n0, dn0] = pooledStepScale(S);
                NSL::size_t next;
                if (s.nBulk < 5 || s.bulkAcceptance() < 0.05 || !std::isfinite(n0)) {
                    next = std::min(2 * cp.nmd, opt.maxNmd);
                } else {
                    const NSL::size_t lo = std::max<NSL::size_t>(1, (cp.nmd + 3) / 4);
                    const NSL::size_t hi = std::max(lo, std::min(2 * cp.nmd, opt.maxNmd));
                    next = std::clamp<NSL::size_t>(NSL::size_t(std::llround(n0 / eStar)), lo, hi);
                }
                NSL::Logger::info("Tuning Nmd {}: {} trajectories, bulk acceptance {:.3f} +- {:.3f}, spikes {:.2f}, "
                                  "step scale N0 = {:.2f} +- {:.2f} (pooled) -> Nmd {}",
                                  cp.nmd, s.n, s.bulkAcceptance(), s.bulkAcceptanceError(), s.spikeFraction(), n0, dn0, next);
                // converged: the proposal is this Nmd, or within the relative tolerance of it, and
                // this Nmd has enough bulk trajectories; when the proposal alternates between two
                // Nmd that both have enough, the latest proposal is taken
                bool converged = false;
                if (s.nBulk >= opt.minTrajectories) {
                    if (next == cp.nmd || std::abs(double(next) - double(cp.nmd)) <= opt.tolerance * double(cp.nmd)) {
                        converged = true;
                    } else if (S.count(next) && S.at(next).nBulk >= opt.minTrajectories) {
                        cp.nmd = next;
                        converged = true;
                    }
                }
                if (converged) {
                    const Stats & f = S.at(cp.nmd);
                    if (f.spikeFraction() > opt.spikeFraction) {
                        cp.compare = 1; cp.low = cp.nmd; cp.high = higher(cp.nmd);
                        NSL::Logger::info("Tuning: {:.0f}% spikes at Nmd {}; comparing Nmd {} and {} on the same chain",
                                          100. * f.spikeFraction(), cp.nmd, cp.low, cp.high);
                    } else {
                        break;
                    }
                } else {
                    cp.nmd = next;   // the same Nmd again if it still needs bulk trajectories
                }
            } else {
                // comparison of low and high, alternating on the same chain
                const int stage = 1 + int(cp.compare);
                const auto S = stats(stage);
                const NSL::size_t nl = S.count(cp.low) ? S.at(cp.low).n : 0;
                const NSL::size_t nh = S.count(cp.high) ? S.at(cp.high).n : 0;
                if (nl >= opt.minTrajectories && nh >= opt.minTrajectories) {
                    const Stats & a = S.at(cp.low);
                    const Stats & b = S.at(cp.high);
                    const double ca = a.cost(cp.low), cb = b.cost(cp.high);
                    NSL::Logger::info("Tuning: Nmd {}: acceptance {:.3f}, spikes {:.2f}, cost {:.3g} MD steps ({:.3g} s/trajectory); "
                                      "Nmd {}: acceptance {:.3f}, spikes {:.2f}, cost {:.3g} MD steps ({:.3g} s/trajectory) "
                                      "(cost per independent configuration)",
                                      cp.low, a.acceptance(), a.spikeFraction(), ca, a.secondsPerTrajectory(),
                                      cp.high, b.acceptance(), b.spikeFraction(), cb, b.secondsPerTrajectory());
                    if (cb < ca && b.spikeFraction() > opt.spikeFraction && cp.compare < 3 && cp.high < opt.maxNmd) {
                        cp.low = cp.high; cp.high = higher(cp.high); ++cp.compare;
                        continue;
                    }
                    cp.nmd = (cb < ca) ? cp.high : cp.low;
                    break;
                }
                cp.nmd = (nl <= nh) ? cp.low : cp.high;
                trajectory(cp.nmd, stage);
                if (stopRequested()) { return false; }
            }
        }
        return true;
    }

    void writeResult(){
        const auto S = stats(1);
        const auto [n0, dn0] = pooledStepScale(S);
        // the statistics at the final Nmd: from the last comparison round if there was one
        Stats fin;
        for (int stage = int(cp.compare) + 1; stage >= 1; --stage) {
            const auto St = stats(stage);
            if (St.count(cp.nmd)) { fin = St.at(cp.nmd); break; }
        }
        NSL::Logger::info("Tuning done: Nmd = {} (bulk acceptance {:.3f}, spikes {:.2f}, step scale N0 = {:.2f} +- {:.2f}); "
                          "{} tuning trajectories", cp.nmd, fin.bulkAcceptance(), fin.spikeFraction(), n0, dn0, tuningTrajectories());
        NSL::Stop::WriteGuard writing;
        const std::string r = recordNode() + "/result";
        h5.deleteData(r);
        h5.write(cp.nmd, r + "/Nmd");
        h5.write(std::isfinite(n0) ? n0 : 0., r + "/N0");
        h5.write(std::isfinite(dn0) ? dn0 : 0., r + "/N0Error");
        h5.write(opt.target, r + "/target");
        h5.write(opt.spikeThreshold, r + "/spikeThreshold");
        h5.write(fin.bulkAcceptance(), r + "/bulkAcceptance");
        h5.write(fin.spikeFraction(), r + "/spikeFraction");
        h5.write(double(tuningTrajectories()), r + "/tuningTrajectories");
    }
};

//! Records the Nmd used in <base>/Meta/params/nMD
inline void recordNmd(NSL::H5IO & h5, const std::string & base, NSL::size_t nmd){
    NSL::Stop::WriteGuard writing;
    const std::string node = base + "/Meta/params/nMD";
    h5.deleteData(node);
    h5.write(nmd, node);
}

} // namespace detail

/*! Thermalization, tuning of Nmd (if tuneNmd) and production of <base>/markovChain with the
 *  RadialHMC `hmc`, starting from `start` (only used for a new chain). Returns EXIT_SUCCESS, or
 *  NSL::Stop::exitCode if it stopped early on a signal (run the same command again to continue).
 */
template<typename HMCType, NSL::Concept::isNumber Type>
int run(HMCType & hmc, const NSL::MCMC::MarkovState<Type> & start, NSL::H5IO & h5, const std::string & base,
        const RunParameters & rp, const Options & opt, bool tuneNmd){
    const std::string chainNode = base + "/markovChain";
    detail::ChainRun<HMCType, Type> c{hmc, h5, base, rp, opt, {}, {}, start};
    const std::string ck = c.checkpointNode();
    NSL::size_t nmd = rp.Nmd;

    if (!h5.exist(chainNode)) {
        if (h5.exist(ck)) {
            h5.read(c.cp.completedSteps, ck + "/completedSteps");
            c.cp.nmd = rp.Nmd;
            if (h5.exist(ck + "/phase")) {
                h5.read(c.cp.phase, ck + "/phase");
                h5.read(c.cp.nmd, ck + "/nmd");
                h5.read(c.cp.compare, ck + "/compare");
                h5.read(c.cp.low, ck + "/low");
                h5.read(c.cp.high, ck + "/high");
            }
            h5.read(c.state, ck + "/state", 0);
            c.record.read(h5, c.recordNode() + "/trajectories");
            NSL::Logger::info("Continuing from the checkpoint in {}: {}, Nmd = {}", ck,
                c.cp.phase == 0 ? fmt::format("thermalization {} of {}", c.cp.completedSteps, rp.Ntherm)
                                : std::string(c.cp.phase == 1 ? "tuning" : "tuning done"), c.cp.nmd);
        } else {
            c.cp.nmd = rp.Nmd;
            h5.deleteData(c.recordNode() + "/trajectories");   // a record of an earlier, discarded attempt
        }
        if (c.cp.phase == 0) {
            if (!c.thermalize(tuneNmd)) { return NSL::Stop::exitCode; }
            c.cp.phase = tuneNmd ? 1 : 2;
            c.writeCheckpoint();
        }
        if (c.cp.phase == 1) {
            if (!c.tune()) { return NSL::Stop::exitCode; }
            c.cp.phase = 2;
            c.writeCheckpoint();
            c.writeResult();
        }
        nmd = c.cp.nmd;
    } else {
        if (h5.exist(c.recordNode() + "/result/Nmd")) { h5.read(nmd, c.recordNode() + "/result/Nmd"); }
        else if (!tuneNmd) { nmd = rp.Nmd; }
        else if (h5.exist(base + "/Meta/params/nMD")) { h5.read(nmd, base + "/Meta/params/nMD"); }
        NSL::Logger::info("Continuing the Markov chain in {} with Nmd = {}", chainNode, nmd);
    }
    if (NSL::Stop::requested()) { return NSL::Stop::exitCode; }

    detail::recordNmd(h5, base, nmd);
    hmc.setNumberSteps(nmd);
    auto gen_time = NSL::Logger::start_profile("Generation");
    NSL::Logger::info("Generating {} configurations with Nmd = {}, saving every {}...", rp.Nconf, nmd, rp.saveFrequency);
    std::vector<NSL::MCMC::MarkovState<Type>> markovChain = hmc.template generate<NSL::MCMC::Chain::AllStates>(
        c.state, rp.Nconf, rp.Nradial, rp.Nhmc, NSL::RealTypeOf<Type>(rp.radialScale), rp.saveFrequency, chainNode
    );
    NSL::Logger::stop_profile(gen_time);
    // the chain starts with the state after thermalization and tuning; the checkpoint is no longer needed
    if (h5.exist(ck)) { h5.deleteData(ck); }
    if (!NSL::Stop::requested()) { diagnose(h5, base, opt); }
    return NSL::Stop::requested() ? NSL::Stop::exitCode : EXIT_SUCCESS;
}

} // namespace NSL::MCMC::Autotune

#endif // NSL_MARKOVCHAIN_AUTOTUNE_TPP
