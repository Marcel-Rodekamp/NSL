/*! \file autotune_RadialMCMC_PPP.cpp
 *  \brief HMC (with radial updates) for the Pariser-Parr-Pople model.
 *
 *  The PPP interaction 1/2 Σ_xy V_xy q_x q_y is decoupled in the charge channel,
 *  so the fermions use the CHARGE (particle/hole) basis matrix HubbardExp together
 *  with NSL::Action::PPPGaugeAction. The interaction matrix V is built from the
 *  `PPP` block of the yml file, see NSL::PPP (src/NSL/Action/Implementations/PPP_Interaction.tpp).
 *
 *  Three modes, selected by `HMC: autotune:` in the yml file (NSL::MCMC::Autotune):
 *   - true (default): `HMC: Ntherm` thermalization updates, tuning of the number of MD
 *     steps (starting at `Leapfrog: Nmd`, default 10) on the thermalized chain, and
 *     `HMC: Nconf` stored configurations with the tuned Nmd, all in one run; see
 *     src/NSL/MarkovChain/autotune.tpp and the optional `Autotune:` block.
 *   - false: a plain run with the fixed `Leapfrog: Nmd`.
 *   - legacy: the protocol of autotuner8/autotuner.py (one trajectory per call during
 *     tuning, Nmd taken from the h5 file); autotuner.py selects it by itself.
 *  `HMC: start: cold` (phi = 0; the default except for a plain leapfrog run) or `gaussian` (a
 *  sample of the gauge action) is the starting field; an optional `HMC: seed` makes a run reproducible. Rerunning on an existing
 *  h5 file continues where the run stopped.
 *
 *  Integrator, selected by `Leapfrog: Fourier acceleration:` (default false):
 *   - false: leapfrog with unit mass (NSL::Integrator::LeapfrogRealForce);
 *   - true:  exact Fourier acceleration (NSL::Integrator::ExactFourierAcceleration):
 *     mass matrix (δV)^{-1}, the gauge action is integrated exactly and only the
 *     fermion force numerically. Use `trajectory length: 1.5707963` (π/2).
 *   `Leapfrog: Nmd` (or the autotuned value) is the number of steps: Nmd+1 fermion
 *   force evaluations per trajectory for the leapfrog, Nmd for Fourier acceleration.
 *
 *  Bridge link (optional; mu = 0, bipartite lattice): `HMC: bridge: {theta: 1.5707963, slice: 0}` samples with
 *  a weight without the zeros of det M_p, which makes the variance of observables quadratic in the propagator
 *  (equal-time correlations, two-body correlators) finite; every measurement must then be reweighted by
 *  1/Lambda, computed from the equal-time propagator at the bridge slice (src/NSL/MarkovChain/bridgeLink.tpp).
 *  `slice: all` is the multi-slice bridge (the bridge averaged over all time slices, Lambda from the equal-time
 *  propagator at every slice): it also bounds the weights of configurations whose near-zero mode avoids slice 0.
 *  The chain records the setting in <name>/Meta/bridge/{theta, slice}, with slice the string "all" for the
 *  multi-slice bridge, and keeps it for life. Without `HMC: bridge:` there is no bridge.
 *
 *  Mode flips (optional): `HMC: mode flips: N` adds N Metropolis proposals per combined update step, after the
 *  radial updates and before the HMC trajectory, each reversing one eigenvector of V on every time slice
 *  (NSL::MCMC::ModeFlip, src/NSL/MarkovChain/modeFlip.tpp). They leave the gauge action unchanged and change the
 *  sign sector of det M_p far more often per unit cost than HMC. For Hubbard (V = U 1) the modes are the sites
 *  (the global moves of Scalettar, Noack and Singh). Exact for every N, with or without the bridge link (with it,
 *  the flips are accepted on the shifted action); no reweighting. 0 or no entry: off, and the chain is the same as
 *  without the option. The setting is recorded in <name>/Meta/modeFlips/{proposals, modes} and may change between
 *  runs of a chain (the target does not); the acceptance is logged at the end of a run.
 *
 *  Measurements: Executables/Hubbard/propagator with `measurements: basis: charge`
 *  (position-space and/or momentum-projected propagators).
 *
 *  Safe early stop (NSL::Stop, e.g. `#SBATCH --signal=USR1@600`): on SIGUSR1 the current
 *  trajectory is finished and stored; thermalization and tuning are checkpointed in
 *  <name>/thermalization and continued by the next run. Exit code 75 means "stopped early,
 *  run again to continue".
 */

#include <chrono>
#include <optional>
#include "NSL.hpp"
#include "highfive/H5File.hpp"
#include <yaml-cpp/yaml.h>

template<NSL::Concept::isNumber Type, typename LatticeType>
void writeMeta(LatticeType lat, NSL::Parameter & params, NSL::H5IO & h5, std::string BASENODE,
               const NSL::PPP::Parameters & ppp, NSL::Tensor<double> V);

static int runPPP(int argc, char* argv[]){
    // Disable autograd graph tracking — gradients are computed analytically
    torch::InferenceMode guard;

    typedef NSL::complex<double> Type;

    // Initialize NSL
    NSL::Parameter params = NSL::init(argc, argv, "PPP HMC");

    auto init_time = NSL::Logger::start_profile("Initialization");

    YAML::Node yml = YAML::LoadFile(params["file"]);

    // The name of the physical system
    params["name"]              = yml["system"]["name"].as<std::string>();
    // The inverse temperature (units of 1/t)
    params["beta"]              = yml["system"]["beta"].as<double>();
    // The number of time slices
    params["Nt"]                = yml["system"]["Nt"].as<NSL::size_t>();
    // The number of ions
    params["Nx"]                = yml["system"]["nions"].as<NSL::size_t>();
    // The on-site interaction (units of t)
    params["U"]                 = yml["system"]["U"].as<double>();
    // The HMC save frequency
    params["save frequency"]    = yml["HMC"]["save frequency"].as<NSL::size_t>();
    // The number of Radial Updates per combined step
    params["Nradial"] = yml["HMC"]["Nradial"] ? yml["HMC"]["Nradial"].as<NSL::size_t>() : NSL::size_t(0);
    // The number of HMC steps per combined step
    params["Nhmc"]    = yml["HMC"]["Nhmc"]    ? yml["HMC"]["Nhmc"].as<NSL::size_t>()    : NSL::size_t(1);
    // The thermalization length
    params["Ntherm"]            = yml["HMC"]["Ntherm"].as<NSL::size_t>();
    // The number of configurations
    params["Nconf"]             = yml["HMC"]["Nconf"].as<NSL::size_t>();
    // The trajectory length
    params["trajectory length"] = yml["Leapfrog"]["trajectory length"].as<double>();
    // The h5 file name to store the simulation results
    params["h5file"]            = yml["fileIO"]["h5file"].as<std::string>();
    // The offset: tangent plane/NLO plane
    params["offset"] = yml["system"]["offset"] ? yml["system"]["offset"].as<double>() : 0.0;
    // Chemical Potential
    params["mu"]     = yml["system"]["mu"]     ? yml["system"]["mu"].as<double>()     : 0.0;

    // tune Nmd (default), plain run, or the legacy protocol of autotuner8/autotuner.py
    using NSL::MCMC::Autotune::Mode;
    using NSL::MCMC::Autotune::Start;
    const Mode mode = NSL::MCMC::Autotune::modeFromYAML(yml);
    const bool legacy = (mode == Mode::Legacy);
    const NSL::MCMC::Autotune::Options tuneOptions = NSL::MCMC::Autotune::Options::fromYAML(yml);

    // exact Fourier acceleration (NSL::Integrator::ExactFourierAcceleration) or plain leapfrog
    const bool fourierAcceleration = yml["Leapfrog"]["Fourier acceleration"]
        ? yml["Leapfrog"]["Fourier acceleration"].as<bool>() : false;
    params["Fourier acceleration"] = fourierAcceleration;

    // starting field: phi = 0 or a sample of the gauge action; the default depends on the
    // integrator and the mode (NSL::MCMC::Autotune::startDefault), the legacy protocol always
    // uses a sample
    const Start start = legacy ? Start::Gaussian
        : NSL::MCMC::Autotune::startFromYAML(yml, NSL::MCMC::Autotune::startDefault(fourierAcceleration, mode == Mode::Tune));
    NSL::Logger::info("Mode: {}; start: {}", NSL::MCMC::Autotune::to_string(mode), start == Start::Cold ? "cold (phi = 0)" : "gaussian");

    // The number of molecular dynamics steps: the starting value of the tuning (default 10),
    // the fixed value of a plain run, or 200 for the legacy protocol (which stores its choice
    // in the h5 file).
    if (legacy) {
        params["Nmd"] = (NSL::size_t) 200;
    } else if (mode == Mode::Plain) {
        if (!yml["Leapfrog"]["Nmd"]) {
            throw std::runtime_error("`HMC: autotune: false` requires `Leapfrog: Nmd`");
        }
        params["Nmd"] = yml["Leapfrog"]["Nmd"].as<NSL::size_t>();
    } else {
        params["Nmd"] = yml["Leapfrog"]["Nmd"] ? yml["Leapfrog"]["Nmd"].as<NSL::size_t>() : NSL::size_t(10);
    }

    // Standard deviation of proposal lognormal distribution in radial udpate
    if (yml["HMC"]["radial scale"]){
        params["radial scale"]    = yml["HMC"]["radial scale"].as<double>();
    } else {
        // DEFAULT: radialScale = 1/sqrt(Volume), the starting value recommended in arXiv:2504.01575
        params["radial scale"]    = 1./std::sqrt(params["Nt"].to<double>()*params["Nx"].to<double>());
    }
    if (params["radial scale"].to<double>()==0.0){
        params["Nradial"] = 0;
    }

    // Stability method
    if (yml["stability"] ) {
      params["stability"]    = yml["stability"].as<std::string>();
    }

    for(auto [key, value]: params){
        if (key == "device" || key == "file") {continue;}
        NSL::Logger::info( "{}: {}", key, value );
    }

    // Flags of the legacy protocol (autotuner.py); the other modes are "tuned" from the start.
    NSL::size_t _thermalFlag = 1;
    params["thermalFlag"] = _thermalFlag;
    NSL::size_t _tuneFlag = legacy ? 0 : 1;
    params["tuneFlag"] = _tuneFlag;

    // create an H5 object to store data
    NSL::H5IO h5(
        params["h5file"].to<std::string>(),
        params["overwrite"].to<bool>() ? NSL::File::Truncate : NSL::File::ReadWrite | NSL::File::OpenOrCreate
    );

    // everything is stored in params["h5file"]/BASENODE/
    std::string BASENODE(fmt::format("{}",std::string(params["name"])));

    // initialize the lattice
    NSL::Lattice::Generic<Type> lattice(yml);
    if (lattice.sites() != params["Nx"].template to<NSL::size_t>()){
        throw std::runtime_error("The number of ions in the parameter file does not match the number of ions in the lattice.");
    }
    if (!lattice.bipartite()) {
        NSL::Logger::warn("The lattice is not bipartite: det(M_p) det(M_h) is complex in general (sign problem).");
    }

    // Put the lattice on the device. (copy to GPU)
    lattice.to(params["device"]);

    // The PPP interaction matrix (units of the hopping)
    const NSL::PPP::Parameters ppp = NSL::PPP::parametersFromYAML(yml);
    NSL::Tensor<double> V = NSL::PPP::interactionMatrixFromYAML(yml);
    NSL::PPP::logSummary(ppp, V);

    // write the meta data to the h5file
    writeMeta<Type,decltype(lattice)>(lattice, params, h5, BASENODE, ppp, V);

    if (legacy) {
        if (h5.exist(fmt::format("{}/Meta/params/nMD",BASENODE))) {
            NSL::size_t temp;
            h5.read(temp, fmt::format("{}/Meta/params/nMD",BASENODE));
            params["Nmd"] = temp;
        }
        h5.read(_thermalFlag, fmt::format("{}/Meta/params/thermalFlag",BASENODE));
        h5.read(_tuneFlag, fmt::format("{}/Meta/params/tuneFlag",BASENODE));
    }

    NSL::Logger::info("Setting up a PPP action with beta={}, Nt={}, U={}, on {}.",
        params["beta"], params["Nt"], params["U"], params["name"]
    );

    // PPP gauge action: S = 1/2 Σ_t φ_t^T (δV)^{-1} φ_t
    NSL::Action::PPPGaugeAction<Type> S_gauge(V, params);

    // fermion action in the CHARGE basis (the field couples as exp(±iφ) to the charge)
    NSL::Action::HubbardFermionAction<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type,decltype(lattice)>
    > S_fermion(lattice,params);

    // set stability method if defined in yml file, otherwise default is "QR"
    if (yml["stability"] ) {
      std::string stabilityMethod = params["stability"];
      S_fermion.hfm_.stabilityMethod = stabilityMethod;// "QR", "DIRECTINVERSE", "SVD"
    }

    // Bridge link (optional, `HMC: bridge:`): heat bath of a field shift before every trajectory, so that the
    // sampled weight has no zeros where det M_p vanishes; measurements are reweighted by 1/Lambda in the
    // analysis (see src/NSL/MarkovChain/bridgeLink.tpp). It must be set up before S_fermion is copied into S.
    using FermionActionType = decltype(S_fermion);
    using BridgeType = NSL::MCMC::BridgeLink<Type, FermionActionType>;
    std::optional<BridgeType> bridge;
    // the slice as written to Meta/bridge/slice and logged: the slice number, or "all"
    auto sliceName = [](NSL::size_t slice) {
        return slice == BridgeType::AllSlices ? std::string("all") : std::to_string(slice);
    };
    if (yml["HMC"]["bridge"]) {
        const YAML::Node b = yml["HMC"]["bridge"];
        const double theta = b["theta"] ? b["theta"].as<double>() : M_PI / 2;
        NSL::size_t slice = 0;
        if (b["slice"]) {
            slice = b["slice"].as<std::string>() == "all" ? BridgeType::AllSlices : b["slice"].as<NSL::size_t>();
        }
        if (!lattice.bipartite() || params["mu"].to<double>() != 0.0 || params["offset"].to<double>() != 0.0) {
            throw std::runtime_error("`HMC: bridge` needs a bipartite lattice, mu = 0 and offset = 0 "
                                     "(there |det M_p|^2 is the sign-free fermion weight)");
        }
        bridge.emplace(S_fermion, params["Nt"].to<NSL::size_t>(), params["Nx"].to<NSL::size_t>(), theta, slice);
        params["bridge theta"] = theta;
        params["bridge slice"] = sliceName(slice);
        NSL::Logger::info("Bridge link: theta = {}, time slice {}; reweight measurements by 1/Lambda", theta,
                          sliceName(slice));
    }

    // Record the bridge settings with the chain: the analysis (ppp_reduce.py) reweights by them. A chain keeps
    // them for its whole life; continuing it with other settings would mix two ensembles in one chain.
    {
        HighFive::File h5file = h5.getFile();
        const std::string node = BASENODE + "/Meta/bridge";
        const bool hasChain = h5.exist(BASENODE + "/thermal") || h5.exist(BASENODE + "/markovChain");
        if (h5.exist(node)) {
            double theta = 0;
            NSL::size_t slice = 0;
            h5file.getDataSet(node + "/theta").read(theta);
            HighFive::DataSet ds = h5file.getDataSet(node + "/slice");
            if (ds.getDataType().getClass() == HighFive::DataTypeClass::String) {
                std::string name;
                ds.read(name);
                if (name != "all") {
                    throw std::runtime_error(fmt::format("{}/slice is \"{}\"; expected a number or \"all\"", node, name));
                }
                slice = BridgeType::AllSlices;
            } else {
                ds.read(slice);
            }
            if (!bridge || theta != bridge->theta() || slice != bridge->slice()) {
                throw std::runtime_error(fmt::format(
                    "{} was generated with the bridge link (theta = {}, slice {}); continue it with the same "
                    "`HMC: bridge` settings", BASENODE, theta, sliceName(slice)));
            }
        } else if (bridge) {
            if (hasChain) {
                throw std::runtime_error(fmt::format(
                    "{} was generated without the bridge link; continuing it with `HMC: bridge` would mix two "
                    "ensembles (start a new chain)", BASENODE));
            }
            const double theta = bridge->theta();
            h5file.createDataSet<double>(node + "/theta", HighFive::DataSpace::From(theta)).write(theta);
            if (bridge->allSlices()) {
                // a string, so that readers that expect a slice number fail instead of using a wrong slice
                const std::string all = "all";
                h5file.createDataSet<std::string>(node + "/slice", HighFive::DataSpace::From(all)).write(all);
            } else {
                const NSL::size_t slice = bridge->slice();
                h5file.createDataSet<NSL::size_t>(node + "/slice", HighFive::DataSpace::From(slice)).write(slice);
            }
        }
    }

    NSL::Action::Action S = S_gauge + S_fermion;

    // Mode flips (optional, `HMC: mode flips: N`): N Metropolis flips of an eigenvector of V per update step, after the
    // radial updates (src/NSL/MarkovChain/modeFlip.tpp). N = 0 is off: no random numbers are drawn.
    const NSL::size_t modeFlips = yml["HMC"]["mode flips"] ? yml["HMC"]["mode flips"].as<NSL::size_t>() : NSL::size_t(0);
    NSL::MCMC::ModeFlip<Type> flips(NSL::MCMC::interactionModes(V), modeFlips);
    params["mode flips"] = modeFlips;
    if (modeFlips > 0) {
        const bool sites = torch::equal(flips.modes(), torch::eye(flips.modes().size(0), torch::kDouble));
        NSL::Logger::info("Mode flips: {} proposals per update step; modes: {}", modeFlips,
                          sites ? "the sites (V is a multiple of the identity)" : "the eigenvectors of V, largest eigenvalue first");
    }
    {
        NSL::Stop::WriteGuard writing;
        const std::string node = BASENODE + "/Meta/modeFlips";
        h5.deleteData(node);
        HighFive::File h5file = h5.getFile();
        h5file.createDataSet<NSL::size_t>(node + "/proposals", HighFive::DataSpace::From(modeFlips)).write(modeFlips);
        NSL::Tensor<double> modes(flips.modes());
        h5.write(modes, node + "/modes");
    }
    // the acceptance of the flips of this run (the counts are shared by the copy inside the HMC)
    auto logFlips = [&flips]() {
        const auto & st = flips.stats();
        if (st.proposed == 0) { return; }
        std::string perMode;
        for (std::size_t k = 0; k < st.proposedPerMode.size(); ++k) {
            perMode += fmt::format(" {:.2f}", st.proposedPerMode[k] ? double(st.acceptedPerMode[k]) / double(st.proposedPerMode[k]) : NAN);
        }
        NSL::Logger::info("Mode flips: {} proposed, acceptance {:.4f}, {} failed; per mode:{}", st.proposed,
                          st.acceptance(), st.failed, perMode);
    };

    // seed (not for the legacy protocol: autotuner.py calls this executable once per
    // trajectory, a fixed seed would repeat the same momenta every time)
    if (yml["HMC"]["seed"]) {
        if (legacy) {
            NSL::Logger::warn("`HMC: seed` is ignored by the legacy protocol.");
        } else {
            NSL::setSeed(yml["HMC"]["seed"].as<NSL::size_t>());
        }
    }

    // Initial configuration: φ = 0 (cold) or a sample of the Gaussian gauge action, φ_t ~ N(0, δV)
    NSL::Configuration<Type> config{
        {"phi",
            NSL::Tensor<Type>(
                NSL::Device(params["device"]),
                NSL::size_t(params["Nt"]),
                NSL::size_t(params["Nx"])
            )
        }
    };
    if (start == Start::Gaussian) {
        S_gauge.sample(config["phi"]);
    } else {
        config["phi"] = Type(0);
    }
    config["phi"].imag() = NSL::RealTypeOf<Type>(params["offset"]);

    // Everything below only depends on the type of the integrator
    auto simulate = [&](auto & integrator, auto preTrajectory) -> int {
        NSL::MCMC::RadialHMC hmc(integrator, S, h5, preTrajectory, NSL::MCMC::NoPostStore{}, flips);
        NSL::Logger::stop_profile(init_time);

        // Wraps the initial configuration into a Markov state (no update is performed).
        // This also defines the tensors of start_state before any h5 read (see issue #160).
        NSL::MCMC::MarkovState<Type> start_state = hmc.template generate<NSL::MCMC::Chain::LastState>(
            config, 1, params["Nradial"], params["Nhmc"], params["radial scale"]
        );

        if (!legacy) {
            // thermalization, tuning of Nmd (mode Tune) and production; resumable
            const NSL::MCMC::Autotune::RunParameters rp{
                params["Nmd"].to<NSL::size_t>(), params["Ntherm"].to<NSL::size_t>(), params["Nconf"].to<NSL::size_t>(),
                params["save frequency"].to<NSL::size_t>(), params["Nradial"].to<NSL::size_t>(), params["Nhmc"].to<NSL::size_t>(),
                params["radial scale"].to<double>()
            };
            const int rc = NSL::MCMC::Autotune::run(hmc, start_state, h5, BASENODE, rp, tuneOptions, mode == Mode::Tune);
            logFlips();
            return rc;
        }

        // ----------------------------------------------------------------------
        // legacy protocol of autotuner8/autotuner.py (as in autotune_RadialMCMC)
        // ----------------------------------------------------------------------
        auto therm_time =  NSL::Logger::start_profile("Thermalization");
        if (_tuneFlag == 0) {
            NSL::size_t n = 2;
            if (h5.exist(fmt::format("{}/thermal",BASENODE))) {
                auto [minConfigID, maxConfigID] = h5.getMinMaxConfigs(fmt::format("{}/thermal",BASENODE));
                n = maxConfigID + 2;
            }

            // We are thermalizing inbetween two stages of autotuning
            if (_thermalFlag == 0) {
                NSL::Logger::info("Thermalizing {} steps...", params["Ntherm"].to<NSL::size_t>());
                h5.read(start_state, BASENODE+"/thermal");
                start_state = hmc.template generate<NSL::MCMC::Chain::LastState>(start_state, params["Ntherm"].to<NSL::size_t>(), params["Nradial"], params["Nhmc"], params["radial scale"]);
            }

            std::vector<NSL::MCMC::MarkovState<Type>> markovChain = hmc.template generate<NSL::MCMC::Chain::AllStates>(start_state, n, params["Nradial"], params["Nhmc"], params["radial scale"], 1, BASENODE+"/thermal");

            return NSL::Stop::requested() ? NSL::Stop::exitCode : EXIT_SUCCESS;  // 75: stopped early on a signal, resumable
        }

        NSL::Logger::info("Appending to previous data.");
        h5.read(start_state, BASENODE+"/thermal");

        NSL::Logger::stop_profile(therm_time);

        auto gen_time =  NSL::Logger::start_profile("Generation");
        NSL::Logger::info("Generating {} steps, saving every {}...", params["Nconf"], params["save frequency"]);
        std::vector<NSL::MCMC::MarkovState<Type>> markovChain = hmc.template generate<NSL::MCMC::Chain::AllStates>(start_state, params["Nconf"], params["Nradial"], params["Nhmc"], params["radial scale"], params["save frequency"], BASENODE+"/markovChain");
        NSL::Logger::stop_profile(gen_time);

        NSL::Logger::info("Acceptance Rate: {}%",
            NSL::MCMC::getAcceptanceRate(markovChain) * 100
        );
        logFlips();

        return NSL::Stop::requested() ? NSL::Stop::exitCode : EXIT_SUCCESS;  // 75: stopped early on a signal, resumable
    }; // simulate

    const double trajectoryLength = params["trajectory length"].to<double>();
    const NSL::size_t Nmd = params["Nmd"];
    if (fourierAcceleration) {
        // the gauge action is integrated exactly with mass matrix (δV)^{-1};
        // only the fermion force is integrated numerically
        NSL::Logger::info("Setting up exact Fourier acceleration with trajectory length {} and {} MD steps.", trajectoryLength, Nmd);
        NSL::Integrator::ExactFourierAcceleration integrator(
            /*action*/S,
            /*quadratic part*/S_gauge,
            /*trajectoryLength*/trajectoryLength,
            /*numberSteps*/Nmd
        );
        return bridge ? simulate(integrator, *bridge) : simulate(integrator, NSL::MCMC::NoPreTrajectory{});
    } else {
        NSL::Logger::info("Setting up a leapfrog integrator with trajectory length {} and {} MD steps.", trajectoryLength, Nmd);
        // equations of motion via the real part of the force
        NSL::Integrator::LeapfrogRealForce integrator(
            /*action*/S,
            /*trajectoryLength*/NSL::RealTypeOf<Type>(trajectoryLength),
            /*numberSteps*/Nmd
        );
        return bridge ? simulate(integrator, *bridge) : simulate(integrator, NSL::MCMC::NoPreTrajectory{});
    }
}

template<NSL::Concept::isNumber Type, typename LatticeType>
void writeMeta(LatticeType lat, NSL::Parameter & params, NSL::H5IO & h5, std::string BASENODE,
               const NSL::PPP::Parameters & ppp, NSL::Tensor<double> V){
    HighFive::File h5file = h5.getFile();

    if(h5.exist(BASENODE+"/Meta")){
        NSL::Logger::info("Meta data already exists, skipping.");
        return;
    }

    auto writeString = [&](const std::string & node, const std::string & value){
        HighFive::DataSet ds = h5file.createDataSet<std::string>(BASENODE+node, HighFive::DataSpace::From(value));
        ds.write(value);
    };
    auto writeComplex = [&](const std::string & node, const Type & value){
        const auto v = static_cast<std::complex<NSL::RealTypeOf<Type>>>(value);
        HighFive::DataSet ds = h5file.createDataSet<std::complex<NSL::RealTypeOf<Type>>>(BASENODE+node, HighFive::DataSpace::From(v));
        ds.write(v);
    };
    auto writeSize = [&](const std::string & node, NSL::size_t value){
        HighFive::DataSet ds = h5file.createDataSet<NSL::size_t>(BASENODE+node, HighFive::DataSpace::From(value));
        ds.write(value);
    };
    auto writeDouble = [&](const std::string & node, double value){
        HighFive::DataSet ds = h5file.createDataSet<double>(BASENODE+node, HighFive::DataSpace::From(value));
        ds.write(value);
    };

    writeString ("/Meta/lattice",            lat.name());
    writeComplex("/Meta/params/U",           Type(params["U"]));
    writeComplex("/Meta/params/beta",        Type(params["beta"]));
    writeSize   ("/Meta/params/Nt",          params["Nt"].to<NSL::size_t>());
    writeSize   ("/Meta/params/spatialDim",  lat.sites());
    writeSize   ("/Meta/params/nMD",         NSL::size_t(params["Nmd"]));
    writeSize   ("/Meta/params/saveFreq",    NSL::size_t(params["save frequency"]));
    writeDouble ("/Meta/params/trajLength",  NSL::RealTypeOf<Type>(params["trajectory length"]));
    writeString ("/Meta/action",             "PPP/hubbardExp");
    writeSize   ("/Meta/params/thermalFlag", NSL::size_t(params["thermalFlag"]));
    writeSize   ("/Meta/params/tuneFlag",    NSL::size_t(params["tuneFlag"]));

    // PPP interaction
    writeString ("/Meta/PPP/form",           NSL::PPP::to_string(ppp.form));
    writeDouble ("/Meta/PPP/hoppingEnergy",  ppp.hoppingEnergy);
    writeDouble ("/Meta/PPP/lengthUnit",     ppp.lengthUnit);
    writeDouble ("/Meta/PPP/dielectric",     ppp.dielectric);
    writeDouble ("/Meta/PPP/e2",             ppp.e2);
    h5.write(V, BASENODE+"/Meta/PPP/V");
}

int main(int argc, char* argv[]){
    // a failure ends the run with a message and a non-zero exit code;
    // everything stored before stays valid
    try {
        return runPPP(argc, argv);
    } catch (const std::exception & e) {
        NSL::Logger::error("autotune_RadialMCMC_PPP: {}", e.what());
        return EXIT_FAILURE;
    }
}
