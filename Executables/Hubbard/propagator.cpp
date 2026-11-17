//! Fermion propagators of a stored ensemble, in the charge or the spin basis.
/*!
 *  propagator -f run.yml [--config N] [--overwrite] [--GPU]
 *
 *  Reads the configurations <name>/markovChain/<n>/phi from fileIO: h5file and writes the
 *  propagators into the same file. Relevant entries of the parameter file:
 *
 *    measurements:
 *      basis: charge | spin       required; the basis the ensemble was generated in:
 *                                 charge = HubbardExp (particles and holes, also PPP),
 *                                 spin   = HubbardExpSpinBasis (spin up and down)
 *      type: spatial | momentum | both
 *                                 optional; default momentum if wallSources are given,
 *                                 otherwise spatial
 *      Number Time Sources: 4     optional, default 1
 *      wallSources: [...]         needed for momentum: blocks x bands x Nx entries [re, im]
 *      momenta: [...]             optional labels of the wall sources, stored with the data
 *    stability: QR | SVD          optional, default QR
 *
 *  --config N starts at configuration N (earlier configurations are not measured),
 *  --overwrite recomputes and replaces existing results (the ensemble is never touched).
 *
 *  HDF5 layout, with <base> = markovChain/<n>, or NonInteracting for phi = 0 (source time 0):
 *    <name>/<base>/propagator/spatial/<species>/invM[t,<s>]   invM[<s>,t]   invM[t,t]
 *    <name>/<base>/propagator/momentum/<species>/invM[t,<s>]/<kSink>-<kSrc>   ...
 *    <name>/Meta/propagator/basis                  "charge" or "spin"
 *    <name>/Meta/propagator/momentum/wallSources   the sources kSink, kSrc refer to
 *    <name>/Meta/propagator/momentum/momenta       the labels, if given
 *    <name>/<base>/propagator/momentum/<species>/Q    1/Nt sum_t Tr invM[t,t]
 *  <species> is particle/hole (charge basis) or spinUp/spinDown (spin basis). Both bases
 *  measure the same quantities; the wall sources are each source of the parameter file
 *  followed by its complex conjugate (kSink, kSrc = 2k and 2k+1).
 */
#include "NSL.hpp"
#include <iostream>
#include <string>
#include <vector>

namespace {

typedef NSL::complex<double> Type;

std::string readString(HighFive::File & file, const std::string & node){
    std::string value;
    file.getDataSet(node).read(value);
    return value;
}

//! Checks the requested basis against what the file records and records it if new
void checkBasis(NSL::H5IO & h5, const std::string & basenode, const std::string & basis){
    HighFive::File & file = h5.getFile();

    // the action the ensemble was generated with (written by the HMC drivers)
    const std::string actionNode = basenode + "/Meta/action";
    if (file.exist(actionNode)) {
        const std::string action = readString(file, actionNode);
        if (basis == "spin" && action.rfind("PPP", 0) == 0) {
            throw std::runtime_error(fmt::format(
                "The ensemble was generated with the action '{}'. PPP ensembles exist only in the charge "
                "basis; use measurements: basis: charge.", action));
        }
        if (basis == "charge" && action.find("SpinBasis") != std::string::npos) {
            throw std::runtime_error(fmt::format(
                "The ensemble was generated in the spin basis (action '{}'); use measurements: basis: spin.", action));
        }
    }

    // propagators measured earlier in this file
    const std::string basisNode = basenode + "/Meta/propagator/basis";
    if (file.exist(basisNode)) {
        const std::string recorded = readString(file, basisNode);
        if (recorded != basis) {
            throw std::runtime_error(fmt::format(
                "{} records propagators measured in the {} basis, but the parameter file asks for the {} "
                "basis. Propagators of one ensemble must use the basis it was generated in.",
                basisNode, recorded, basis));
        }
    } else {
        file.createDataSet<std::string>(basisNode, HighFive::DataSpace::From(basis)).write(basis);
    }
}

//! Runs the requested measurements with the propagator class of the chosen basis
template<typename Propagator, typename LatticeType>
void run(LatticeType & lattice, NSL::Parameter & params, NSL::H5IO & h5,
         const std::string & stability, bool spatial, bool momentum, NSL::size_t startConfig){
    Propagator invM(lattice, params, h5);
    invM.hfm_.stabilityMethod = stability;

    // momentum first: it checks the wall sources against those already in the file
    if (momentum) { invM.measureK(startConfig); }
    if (spatial)  { invM.measure(startConfig); }
}

} // namespace

int main(int argc, char* argv[]){
    torch::InferenceMode guard;

    NSL::Parameter params = NSL::init(argc, argv, "propagator");
    YAML::Node yml = YAML::LoadFile(params["file"]);

    // system
    params["name"] = yml["system"]["name"].as<std::string>();
    params["beta"] = yml["system"]["beta"].as<double>();
    params["Nt"]   = yml["system"]["Nt"].as<NSL::size_t>();
    params["Nx"]   = yml["system"]["nions"].as<NSL::size_t>();
    params["mu"]   = yml["system"]["mu"] ? yml["system"]["mu"].as<double>() : 0.0;
    params["h5file"] = yml["fileIO"]["h5file"].as<std::string>();

    // what to measure
    const YAML::Node meas = yml["measurements"];
    if (!meas || !meas["basis"]) {
        throw std::runtime_error(
            "measurements: basis: is required (charge for HubbardExp and PPP ensembles, "
            "spin for HubbardExpSpinBasis ensembles)");
    }
    const std::string basis = meas["basis"].as<std::string>();
    if (basis != "charge" && basis != "spin") {
        throw std::runtime_error(fmt::format("measurements: basis: must be charge or spin, not '{}'", basis));
    }

    const bool haveWallSources = static_cast<bool>(meas["wallSources"]);
    const std::string type = meas["type"] ? meas["type"].as<std::string>() : (haveWallSources ? "momentum" : "spatial");
    if (type != "spatial" && type != "momentum" && type != "both") {
        throw std::runtime_error(fmt::format("measurements: type: must be spatial, momentum or both, not '{}'", type));
    }
    const bool spatial  = (type == "spatial"  || type == "both");
    const bool momentum = (type == "momentum" || type == "both");
    if (momentum && !haveWallSources) {
        throw std::runtime_error("measurements: type: " + type + " needs measurements: wallSources:");
    }

    params["Number Time Sources"] = meas["Number Time Sources"] ? meas["Number Time Sources"].as<NSL::size_t>() : NSL::size_t(1);

    // stability of the propagator computation; the propagators implement QR and SVD
    const std::string stability = yml["stability"] ? yml["stability"].as<std::string>() : "QR";
    if (stability != "QR" && stability != "SVD") {
        throw std::runtime_error(fmt::format("stability: must be QR or SVD for propagators, not '{}'", stability));
    }
    params["stability"] = stability;

    // first configuration to measure
    int startConfig = 0;
    if (params["config"]) {
        startConfig = params["config"];
    }
    if (startConfig < 0) {
        throw std::runtime_error(fmt::format("--config must not be negative, got {}", startConfig));
    }

    // lattice
    NSL::Lattice::Generic<Type> lattice(yml);
    if (lattice.sites() != params["Nx"].template to<NSL::size_t>()){
        throw std::runtime_error("The number of ions in the parameter file does not match the number of ions in the lattice.");
    }
    lattice.to(params["device"]);

    // wall sources (blocks x bands x Nx, complex) and their optional momentum labels
    if (haveWallSources) {
        const auto ws = meas["wallSources"].as<std::vector<std::vector<std::vector<std::vector<double>>>>>();
        const NSL::size_t blocks = ws.size(), bands = ws.at(0).size(), Nx = ws.at(0).at(0).size();
        if (Nx != lattice.sites()) {
            throw std::runtime_error(fmt::format("wallSources have {} entries per band, the lattice has {} sites", Nx, lattice.sites()));
        }
        NSL::Tensor<Type> w(blocks, bands, Nx);
        for (NSL::size_t i = 0; i < blocks; ++i) {
            if (ws[i].size() != bands) { throw std::runtime_error("all wallSources blocks must have the same number of bands"); }
            for (NSL::size_t j = 0; j < bands; ++j) {
                if (ws[i][j].size() != Nx) { throw std::runtime_error("all wallSources bands must have one entry per site"); }
                for (NSL::size_t k = 0; k < Nx; ++k) {
                    w(i, j, k) = Type(ws[i][j][k].at(0), ws[i][j][k].at(1));
                }
            }
        }
        NSL::Logger::info("Wall sources: {} block(s), {} band(s), {} sites", blocks, bands, Nx);
        params["wallSources"] = w.to(params["device"]);

        if (meas["momenta"]) {
            const auto m = meas["momenta"].as<std::vector<std::vector<double>>>();
            NSL::Tensor<double> momenta(m.size(), m.at(0).size());
            for (NSL::size_t i = 0; i < m.size(); ++i) {
                for (NSL::size_t j = 0; j < m[i].size(); ++j) {
                    momenta(i, j) = m[i][j];
                }
            }
            params["momenta"] = momenta;
        }
    }

    NSL::Logger::info("Measuring {} propagators in the {} basis (stability {}), from configuration {}",
        type, basis, stability, startConfig);

    // the ensemble file must exist and is never truncated; --overwrite only replaces results
    NSL::H5IO h5(params["h5file"].to<std::string>(), NSL::File::ReadWrite);
    checkBasis(h5, std::string(params["name"]), basis);

    if (basis == "charge") {
        run<NSL::Measure::Hubbard::FermionPropagator<
            Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type, decltype(lattice)>
        >>(lattice, params, h5, stability, spatial, momentum, NSL::size_t(startConfig));
    } else {
        run<NSL::Measure::Hubbard::FermionPropagatorSpinBasis<
            Type, decltype(lattice), NSL::FermionMatrix::HubbardExpSpinBasis<Type, decltype(lattice)>
        >>(lattice, params, h5, stability, spatial, momentum, NSL::size_t(startConfig));
    }

    return EXIT_SUCCESS;
}
