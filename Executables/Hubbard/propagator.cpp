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
 *      type: spatial              optional; one of spatial, momentum, eigenbasis, or a list of
 *                                 them (both = [spatial, momentum]). Default: momentum if
 *                                 wallSources are given, eigenbasis if eigenvalueIndices or
 *                                 fermiWindow are given, otherwise spatial.
 *      Number Time Sources: 4     optional, default 1
 *      wallSources: [...]         momentum: blocks x bands x Nx entries [re, im]
 *      momenta: [...]             optional labels of the wall sources, stored with the data
 *      fermiWindow: {below: 1, above: 0}
 *                                 eigenbasis: the levels HOMO-below .. LUMO+above (defaults)
 *      eigenvalueIndices: [...]   eigenbasis: explicit energy indices instead of the window
 *    stability: ...               propagators are always measured with QR (see below)
 *
 *  --config N starts at configuration N (earlier configurations are not measured),
 *  --overwrite recomputes and replaces existing results (the ensemble is never touched).
 *
 *  HDF5 layout, with <base> = markovChain/<n>, or NonInteracting for phi = 0 (source time 0):
 *    <name>/<base>/propagator/spatial/<species>/invM[t,<s>]   invM[<s>,t]   invM[t,t]
 *    <name>/<base>/propagator/momentum/<species>/invM[t,<s>]/<kSink>-<kSrc>   ...,  Q
 *    <name>/<base>/propagator/eigenbasis/<species>/invM[t,<s>]   ...,  Q
 *    <name>/Meta/propagator/basis                  "charge" or "spin"
 *    <name>/Meta/propagator/momentum/wallSources   the sources kSink, kSrc refer to
 *    <name>/Meta/propagator/momentum/momenta       the labels, if given
 *    <name>/Meta/propagator/eigenbasis/...         the selected hopping eigenvectors
 *  <species> is particle/hole (charge basis) or spinUp/spinDown (spin basis); Q = 1/Nt sum_t
 *  Tr invM[t,t]. Both bases measure the same quantities. Momentum: each wall source followed by
 *  its complex conjugate (kSink, kSrc = 2k and 2k+1). Eigenbasis: eigenvectors v_n of the hopping
 *  matrix with single-particle energies eps_n = -(lambda_n + mu), selected around the Fermi
 *  level of the non-interacting filling (see PropagatorTools::selectHoppingEigenbasis); one
 *  dataset (Nt, m, m) per component.
 *
 *  Stability: the SVD path of the propagator measurement loses accuracy at low temperature
 *  (relative error 6e-4 for benzene at beta = 20, non-finite values at beta = 90) and there is
 *  no DIRECTINVERSE path, so the measurement always uses QR; another `stability:` in the
 *  parameter file (used by the HMC) only gives a warning.
 */
#include "NSL.hpp"
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace {

typedef NSL::complex<double> Type;

std::string readString(HighFive::File & file, const std::string & node){
    std::string value;
    file.getDataSet(node).read(value);
    return value;
}

//! A key that is missing, null, or an empty list or map counts as absent
bool present(const YAML::Node & node){
    return node && !node.IsNull() && !((node.IsSequence() || node.IsMap()) && node.size() == 0);
}

/*! A non-negative YAML integer: a plain (unquoted) scalar or one tagged !!int, made of decimal
 *  digits with an optional sign. Floats (1.0, 1e3), strings ("2"), booleans, lists and numbers
 *  with a leading zero (010 is octal in YAML 1.1) are errors. Values above INT_MAX saturate.
 */
int nonNegativeInt(const YAML::Node & val, const std::string & what){
    const std::string text = val.IsScalar() ? val.Scalar() : std::string();
    const std::string tag = val.IsScalar() ? val.Tag() : std::string();
    std::string digits = text;
    bool negative = false;
    if (!digits.empty() && (digits[0] == '+' || digits[0] == '-')) { negative = digits[0] == '-'; digits.erase(0, 1); }
    const bool isInt = (tag == "?" || tag == "tag:yaml.org,2002:int")
        && !digits.empty() && digits.find_first_not_of("0123456789") == std::string::npos;
    if (!isInt) {
        throw std::runtime_error(fmt::format("{} must be a non-negative integer (got {})", what,
            val.IsNull() ? std::string("null") : val.IsSequence() ? std::string("a list") : val.IsMap() ? std::string("a map")
            : tag == "!" ? "the string \"" + text + "\"" : "'" + text + "'"));
    }
    if (digits.size() > 1 && digits[0] == '0') {
        throw std::runtime_error(fmt::format("{} has a leading zero ({}; YAML 1.1 reads it as octal); write it without", what, text));
    }
    if (negative && digits.find_first_not_of('0') != std::string::npos) {
        throw std::runtime_error(fmt::format("{} must be a non-negative integer (got {})", what, text));
    }
    const std::size_t nz = digits.find_first_not_of('0');
    const std::string sig = nz == std::string::npos ? std::string("0") : digits.substr(nz);
    const int intMax = std::numeric_limits<int>::max();
    return (sig.size() > 10 || std::stoll(sig) > intMax) ? intMax : int(std::stoll(sig));
}

//! measurements: wallSources (blocks x bands x Nx entries [re, im]); absent if missing or empty
std::optional<NSL::Tensor<Type>> parseWallSources(const YAML::Node & node, NSL::size_t Nx){
    if (!present(node)) {
        if (node && !node.IsNull()) { NSL::Logger::warn("measurements: wallSources is empty, treated as absent"); }
        return std::nullopt;
    }
    const auto ws = node.as<std::vector<std::vector<std::vector<std::vector<double>>>>>();
    const NSL::size_t blocks = ws.size(), bands = ws.at(0).size();
    bool wellFormed = bands > 0;
    for (const auto & block: ws) {
        wellFormed = wellFormed && block.size() == bands;
        for (const auto & vec: block) {
            wellFormed = wellFormed && NSL::size_t(vec.size()) == Nx;
            for (const auto & z: vec) { wellFormed = wellFormed && z.size() == 2; }
        }
    }
    if (!wellFormed) {
        throw std::runtime_error(fmt::format(
            "measurements: wallSources must be blocks x bands x Nx ({}) entries [re, im], with the same number (> 0) of bands in every block", Nx));
    }
    NSL::Tensor<Type> w(blocks, bands, Nx);
    for (NSL::size_t i = 0; i < blocks; ++i) {
        for (NSL::size_t j = 0; j < bands; ++j) {
            for (NSL::size_t k = 0; k < Nx; ++k) { w(i, j, k) = Type(ws[i][j][k][0], ws[i][j][k][1]); }
        }
    }
    NSL::Logger::info("Wall sources: {} block(s), {} band(s), {} sites", blocks, bands, Nx);
    return w;
}

//! measurements: eigenvalueIndices: distinct energy indices in [0, Nx); absent if missing or empty
std::optional<std::vector<int64_t>> parseEigenvalueIndices(const YAML::Node & node, NSL::size_t Nx){
    if (!present(node)) { return std::nullopt; }
    if (!node.IsSequence()) { throw std::runtime_error("measurements: eigenvalueIndices must be a list of integers"); }
    std::vector<int64_t> idx;
    std::set<int64_t> seen;
    for (const auto & v: node) {
        const int64_t i = nonNegativeInt(v, "each of measurements: eigenvalueIndices");
        if (i >= Nx) { throw std::runtime_error(fmt::format("measurements: eigenvalueIndices: {} is not in [0,{})", i, Nx)); }
        if (!seen.insert(i).second) { throw std::runtime_error(fmt::format("measurements: eigenvalueIndices: {} is listed twice", i)); }
        idx.push_back(i);
    }
    return idx;
}

//! measurements: fermiWindow: {below, above}; missing keys keep the defaults 1 and 0
std::pair<int, int> parseFermiWindow(const YAML::Node & node){
    std::pair<int, int> window{1, 0};
    if (!present(node)) { return window; }
    if (!node.IsMap()) { throw std::runtime_error("measurements: fermiWindow must be a map {below: <int >= 0>, above: <int >= 0>}"); }
    std::set<std::string> seen;
    for (const auto & kv: node) {
        const std::string key = kv.first.IsScalar() ? kv.first.Scalar() : std::string("<non-scalar>");
        if (key != "below" && key != "above") {
            throw std::runtime_error(fmt::format("measurements: fermiWindow has an unknown key '{}' (allowed: below, above)", key));
        }
        if (!seen.insert(key).second) { throw std::runtime_error(fmt::format("measurements: fermiWindow: '{}' is given twice", key)); }
        (key == "below" ? window.first : window.second) = nonNegativeInt(kv.second, "measurements: fermiWindow: " + key);
    }
    return window;
}

//! measurements: type: one name or a list of spatial, momentum, eigenbasis (both = spatial + momentum)
std::set<std::string> parseType(const YAML::Node & node, bool wallSources, bool eigenbasisKeys){
    std::set<std::string> blocks;
    if (!present(node)) {
        if (wallSources) { blocks.insert("momentum"); }
        if (eigenbasisKeys) { blocks.insert("eigenbasis"); }
        if (blocks.empty()) { blocks.insert("spatial"); }
        return blocks;
    }
    std::vector<std::string> names = node.IsSequence() ? node.as<std::vector<std::string>>() : std::vector<std::string>{node.as<std::string>()};
    for (const std::string & name: names) {
        if (name == "both") { blocks.insert("spatial"); blocks.insert("momentum"); }
        else if (name == "spatial" || name == "momentum" || name == "eigenbasis") { blocks.insert(name); }
        else { throw std::runtime_error(fmt::format("measurements: type: '{}' is not spatial, momentum, eigenbasis or both", name)); }
    }
    return blocks;
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
         const std::set<std::string> & blocks, NSL::size_t startConfig){
    Propagator invM(lattice, params, h5);
    invM.hfm_.stabilityMethod = "QR";

    // projections first: they check their sources against those already in the file
    if (blocks.contains("momentum"))   { invM.measureK(startConfig); }
    if (blocks.contains("eigenbasis")) { invM.measureEigenbasis(startConfig); }
    if (blocks.contains("spatial"))    { invM.measure(startConfig); }
}

int propagator(NSL::Parameter & params){
    YAML::Node yml = YAML::LoadFile(params["file"]);

    // system
    params["name"] = yml["system"]["name"].as<std::string>();
    params["beta"] = yml["system"]["beta"].as<double>();
    params["Nt"]   = yml["system"]["Nt"].as<NSL::size_t>();
    params["Nx"]   = yml["system"]["nions"].as<NSL::size_t>();
    params["mu"]   = yml["system"]["mu"] ? yml["system"]["mu"].as<double>() : 0.0;
    params["h5file"] = yml["fileIO"]["h5file"].as<std::string>();
    const NSL::size_t Nx = params["Nx"].to<NSL::size_t>();

    // basis
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

    // what to measure
    const std::optional<NSL::Tensor<Type>> wallSources = parseWallSources(meas["wallSources"], Nx);
    const std::optional<std::vector<int64_t>> eigenvalueIndices = parseEigenvalueIndices(meas["eigenvalueIndices"], Nx);
    const bool fermiWindowGiven = present(meas["fermiWindow"]);
    const auto [below, above] = parseFermiWindow(meas["fermiWindow"]);
    const std::set<std::string> blocks = parseType(meas["type"], wallSources.has_value(), eigenvalueIndices.has_value() || fermiWindowGiven);

    if (blocks.contains("momentum") && !wallSources) {
        throw std::runtime_error("measurements: type: momentum needs measurements: wallSources:");
    }
    if (blocks.contains("eigenbasis")) {
        if (eigenvalueIndices) {
            NSL::Tensor<double> idx(eigenvalueIndices->size());
            for (size_t i = 0; i < eigenvalueIndices->size(); ++i) { idx(i) = double((*eigenvalueIndices)[i]); }
            params["eigenvalueIndices"] = idx;
            if (fermiWindowGiven) { NSL::Logger::warn("measurements: fermiWindow is ignored because eigenvalueIndices are given"); }
        } else {
            params["fermiWindowBelow"] = below;
            params["fermiWindowAbove"] = above;
        }
    } else if (eigenvalueIndices || fermiWindowGiven) {
        NSL::Logger::warn("measurements: eigenvalueIndices/fermiWindow are ignored because type does not include eigenbasis");
    }

    params["Number Time Sources"] = meas["Number Time Sources"] ? meas["Number Time Sources"].as<NSL::size_t>() : NSL::size_t(1);

    // the propagators are measured with QR, whatever the HMC uses
    if (yml["stability"] && yml["stability"].as<std::string>() != "QR") {
        NSL::Logger::warn("stability: {} applies to the HMC only; propagators are measured with QR "
            "(their SVD path loses accuracy at low temperature, and there is no DIRECTINVERSE path)",
            yml["stability"].as<std::string>());
    }

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
    if (lattice.sites() != Nx){
        throw std::runtime_error("The number of ions in the parameter file does not match the number of ions in the lattice.");
    }
    lattice.to(params["device"]);

    if (wallSources) {
        params["wallSources"] = NSL::Tensor<Type>(*wallSources).to(params["device"]);
        if (present(meas["momenta"])) {
            const auto m = meas["momenta"].as<std::vector<std::vector<double>>>();
            NSL::Tensor<double> momenta(m.size(), m.at(0).size());
            for (size_t i = 0; i < m.size(); ++i) {
                for (size_t j = 0; j < m[i].size(); ++j) { momenta(i, j) = m[i][j]; }
            }
            params["momenta"] = momenta;
        }
    }

    std::string list;
    for (const std::string & b: blocks) { list += (list.empty() ? "" : ", ") + b; }
    NSL::Logger::info("Measuring {} propagators in the {} basis, from configuration {}", list, basis, startConfig);

    // the ensemble file must exist and is never truncated; --overwrite only replaces results
    NSL::H5IO h5(params["h5file"].to<std::string>(), NSL::File::ReadWrite);
    checkBasis(h5, std::string(params["name"]), basis);

    if (basis == "charge") {
        run<NSL::Measure::Hubbard::FermionPropagator<
            Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type, decltype(lattice)>
        >>(lattice, params, h5, blocks, NSL::size_t(startConfig));
    } else {
        run<NSL::Measure::Hubbard::FermionPropagatorSpinBasis<
            Type, decltype(lattice), NSL::FermionMatrix::HubbardExpSpinBasis<Type, decltype(lattice)>
        >>(lattice, params, h5, blocks, NSL::size_t(startConfig));
    }
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char* argv[]){
    torch::InferenceMode guard;
    NSL::Parameter params = NSL::init(argc, argv, "propagator");

    // errors (parameter file, basis or sources that contradict the file, non-finite results)
    // end the run with a message; results already written stay valid
    try {
        return propagator(params);
    } catch (const std::exception & e) {
        NSL::Logger::error("propagator: {}", e.what());
        return EXIT_FAILURE;
    }
}
