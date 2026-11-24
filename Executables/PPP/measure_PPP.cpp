//! Observables of a stored PPP (or Hubbard, charge basis) ensemble, measured from its fields: no propagators are written.
/*!
 *  measure_PPP -f run.yml
 *
 *  Reads the configurations <name>/markovChain/<n> (phi, actVal, acceptanceProbability, markovTime) of
 *  fileIO: h5file, opened read-only, and appends their observables (NSL::Measure::Hubbard::ReducedObservables) to
 *  the stage-1 file (the content of a file of Executables/PPP/python/ppp_reduce.py) and, if it exists, to the
 *  stage-2 file (that of ppp_reduce2.py). The stage-1 file must exist; both are written, empty, by
 *      python Executables/PPP/python/ppp_reduce.py  run.yml --skeleton --bin B [--single] [--probe-slices ...]
 *      python Executables/PPP/python/ppp_reduce2.py run.yml --skeleton --bin B [--single] [--window-energy ...]
 *  which also fix the bin size (the same in both), the precision, the probe slices, the orbital basis and the
 *  operator bases. Their paths are reduced/<yml name>.h5 and reduced2/<yml name>.h5 (relative to the working
 *  directory, the defaults of the Python reductions), or `measurements: reduced: {stage1: ..., stage2: ...}` of
 *  the yml.
 *
 *  Relevant entries of the yml: system (name, beta, Nt, nions, mu, lattice), fileIO: h5file,
 *  measurements: Number Time Sources (default 1, as for Executables/Hubbard/propagator); measurements: basis, if
 *  given, must be charge.
 *
 *  Configurations are measured from the first one not yet in the file, and only as many as fill complete bins: a
 *  remainder after the last complete bin is left for a later run. Safe stop (SIGUSR1, NSL::Stop): the current
 *  configuration is finished, the bins written so far stay, exit code 75; run again to continue.
 */
#include "NSL.hpp"
#include <filesystem>
#include <string>

namespace {

typedef NSL::complex<double> Type;

//! the stage-n file of a run (n = 1, 2): measurements: reduced: stage<n>: of the yml, or reduced/<yml name>.h5,
//! reduced2/<yml name>.h5
std::string stagePath(const YAML::Node & yml, const std::string & ymlFile, int n){
    const YAML::Node meas = yml["measurements"];
    const std::string key = "stage" + std::to_string(n);
    if (meas && meas["reduced"] && meas["reduced"][key]) {
        return meas["reduced"][key].as<std::string>();
    }
    return (n == 1 ? "reduced/" : "reduced2/") + std::filesystem::path(ymlFile).stem().string() + ".h5";
}

int measurePPP(NSL::Parameter & params){
    const std::string ymlFile = params["file"];
    YAML::Node yml = YAML::LoadFile(ymlFile);

    params["name"] = yml["system"]["name"].as<std::string>();
    params["beta"] = yml["system"]["beta"].as<double>();
    params["Nt"]   = yml["system"]["Nt"].as<NSL::size_t>();
    params["Nx"]   = yml["system"]["nions"].as<NSL::size_t>();
    params["mu"]   = yml["system"]["mu"] ? yml["system"]["mu"].as<double>() : 0.0;
    params["h5file"] = yml["fileIO"]["h5file"].as<std::string>();
    const std::string name = params["name"];
    const NSL::size_t Nt = params["Nt"].to<NSL::size_t>();
    const NSL::size_t Nx = params["Nx"].to<NSL::size_t>();

    const YAML::Node meas = yml["measurements"];
    if (meas && meas["basis"] && meas["basis"].as<std::string>() != "charge") {
        throw std::runtime_error("measure_PPP measures charge-basis (HubbardExp, PPP) ensembles; measurements: basis: is "
                                 + meas["basis"].as<std::string>());
    }
    params["Number Time Sources"] = (meas && meas["Number Time Sources"])
        ? meas["Number Time Sources"].as<NSL::size_t>() : NSL::size_t(1);

    NSL::Lattice::Generic<Type> lattice(yml);
    if (lattice.sites() != Nx) {
        throw std::runtime_error("The number of ions in the parameter file does not match the number of ions in the lattice.");
    }
    lattice.to(params["device"]);

    const std::string stage1 = stagePath(yml, ymlFile, 1);
    std::string stage2 = stagePath(yml, ymlFile, 2);
    if (!std::filesystem::exists(stage2)) {
        NSL::Logger::info("{} does not exist: stage 1 only (write it with ppp_reduce2.py {} --skeleton for stage 2)", stage2, ymlFile);
        stage2.clear();
    }
    if (!std::filesystem::exists(stage1)) {
        throw std::runtime_error(fmt::format("{} does not exist; write it first with\n  python ppp_reduce.py {} --skeleton "
            "--bin B [--single] [--probe-slices ...] [-o {}]", stage1, ymlFile, stage1));
    }
    NSL::H5IO chain(params["h5file"].to<std::string>(), NSL::File::ReadOnly);
    NSL::Measure::Hubbard::ReducedObservables<Type, decltype(lattice)> obs(lattice, params, chain, stage1, stage2);

    // the stored configurations that complete bins
    const std::string base = name + "/markovChain";
    auto [minCfg, maxCfg] = chain.getMinMaxConfigs(base);
    const NSL::size_t B = obs.binSize();
    const NSL::size_t end = ((NSL::size_t(maxCfg) + 1) / B) * B;      // ids next() .. end - 1
    if (obs.next() >= end) {
        NSL::Logger::info("Nothing to measure: {} has {} bins; the chain has configurations up to {} (bins of {})",
            stage1, obs.bins(), maxCfg, B);
        return EXIT_SUCCESS;
    }
    NSL::Logger::info("Measuring configurations {} to {} of {} ({} bins of {}) into {}",
        obs.next(), end - 1, params["h5file"].to<std::string>(), (end - obs.next()) / B, B,
        stage2.empty() ? stage1 : stage1 + " and " + stage2);

    NSL::Configuration<Type> config{{"phi", NSL::Tensor<Type>(Nt, Nx)}};
    NSL::MCMC::MarkovState<Type> state(config, Type(0), 0., 0, true);
    auto start = std::chrono::steady_clock::now(), last = start;
    for (NSL::size_t id = obs.next(); id < end; ++id) {
        if (NSL::Stop::requested()) {
            NSL::Logger::warn("measure_PPP: stopped ({}) before configuration {}; {} has {} complete bins. Run again to continue.",
                NSL::Stop::signalName(), id, stage1, obs.bins());
            return NSL::Stop::exitCode;
        }
        const std::string node = fmt::format("{}/{}", base, id);
        if (!chain.exist(node + "/phi")) {
            throw std::runtime_error(fmt::format("{} is missing: configuration ids must be 0, 1, 2, ...", node));
        }
        chain.read(state, base, int(id));
        int markovTime = 0;
        chain.read(markovTime, node + "/markovTime");
        obs.add(id, state.configuration["phi"], state.actionValue, double(state.acceptanceProbability), double(markovTime));

        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last).count() > 60 || id + 1 == end) {
            last = now;
            NSL::Logger::info("measure_PPP: configuration {}/{}, {} bins, {:.0f} s", id, end - 1, obs.bins(),
                std::chrono::duration<double>(now - start).count());
        }
    }
    if (obs.holeRelationDeviation() > 1e-8) {
        NSL::Logger::warn("particle-hole relation violated by {:.1e} on the first configurations (expected ~1e-12 at mu = 0 "
            "on a bipartite lattice)", obs.holeRelationDeviation());
    }
    if (obs.stage2() && obs.contractionDeviation() > 1e-8) {
        NSL::Logger::warn("pair contractions deviate from the general ones by {:.1e} on the first configurations",
            obs.contractionDeviation());
    }
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char* argv[]){
    torch::InferenceMode guard;
    NSL::Parameter params = NSL::init(argc, argv, "measure_PPP");
    try {
        return measurePPP(params);
    } catch (const std::exception & e) {
        NSL::Logger::error("measure_PPP: {}", e.what());
        return EXIT_FAILURE;
    }
}
