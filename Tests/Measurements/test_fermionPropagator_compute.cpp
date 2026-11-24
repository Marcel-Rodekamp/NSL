#include "../test.hpp"
#include <filesystem>

// NSL::Measure::Hubbard::FermionPropagator::compute (src/NSL/Measurements/Hubbard/fermionPropagator.tpp) returns
// in memory the spatial propagators that measure() writes to the h5 file: the same values, for both species, on a
// 4-site ring; also for source times without 0 (the diagonal then still uses source time 0, as in measure()).

namespace {
typedef NSL::complex<double> Type;

double maxAbsDiff(NSL::Tensor<Type> a, NSL::Tensor<Type> b){
    return (torch::Tensor(a) - torch::Tensor(b)).abs().max().template item<double>();
}
} // namespace

TEST_CASE("FermionPropagator: compute equals measure", "[Measurements,propagator]") {
    const NSL::size_t Nt = 8, Nx = 4, Nsrc = 3;            // source times 0, 3, 6 (step ceil(8/3) = 3)
    NSL::Lattice::Ring<Type> lattice(Nx);
    NSL::Parameter params;
    params["name"] = std::string("ring");
    params["beta"] = 2.;
    params["Nt"] = Nt;
    params["Nx"] = Nx;
    params["mu"] = 0.;
    params["device"] = NSL::Device("CPU");
    params["overwrite"] = false;
    params["Number Time Sources"] = Nsrc;

    const std::string file = "test_fermionPropagator_compute.h5";
    std::filesystem::remove(file);
    NSL::H5IO h5(file, NSL::File::Truncate);

    NSL::Tensor<Type> phi(Nt, Nx);
    phi.randn();
    phi.imag() = 0;
    phi *= 0.8;
    h5.write(phi, "ring/markovChain/0/phi");

    using Propagator = NSL::Measure::Hubbard::FermionPropagator<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type, decltype(lattice)>>;
    Propagator invM(lattice, params, h5);
    invM.hfm_.stabilityMethod = "QR";
    invM.measure(0);

    const std::vector<NSL::size_t> sources = Propagator::sourceTimes(Nt, Nsrc);
    REQUIRE(sources == std::vector<NSL::size_t>{0, 3, 6});

    NSL::Tensor<Type> stored(Nt, Nx, Nx);
    auto read = [&](const std::string & species, const std::string & component){
        h5.read(stored, fmt::format("ring/markovChain/0/propagator/spatial/{}/{}", species, component));
        return NSL::Tensor<Type>(stored, true);
    };

    for (const NSL::Hubbard::Species species: {NSL::Hubbard::Particle, NSL::Hubbard::Hole}) {
        const std::string name = species == NSL::Hubbard::Particle ? "particle" : "hole";
        INFO("species " << name);
        const auto out = invM.compute(phi, species, sources);
        REQUIRE(out.column.size() == sources.size());
        REQUIRE(out.row.size() == sources.size());
        for (std::size_t i = 0; i < sources.size(); ++i) {
            INFO("source time " << sources[i]);
            CHECK(maxAbsDiff(out.column[i], read(name, fmt::format("invM[t,{}]", sources[i]))) == 0.);
            CHECK(maxAbsDiff(out.row[i], read(name, fmt::format("invM[{},t]", sources[i]))) == 0.);
        }
        CHECK(maxAbsDiff(out.diagonal, read(name, "invM[t,t]")) == 0.);
    }

    // a source list without 0
    const auto single = invM.compute(phi, NSL::Hubbard::Particle, {3});
    CHECK(maxAbsDiff(single.column[0], read("particle", "invM[t,3]")) == 0.);
    CHECK(maxAbsDiff(single.row[0], read("particle", "invM[3,t]")) == 0.);
    CHECK(maxAbsDiff(single.diagonal, read("particle", "invM[t,t]")) == 0.);

    // the results are copies: a later computation does not change them
    const auto first = invM.compute(phi, NSL::Hubbard::Particle, {0});
    NSL::Tensor<Type> keep(first.column[0], true);
    NSL::Tensor<Type> other(Nt, Nx);
    other.randn();
    other.imag() = 0;
    invM.compute(other, NSL::Hubbard::Particle, {0});
    CHECK(maxAbsDiff(first.column[0], keep) == 0.);

    REQUIRE_THROWS(invM.compute(phi, NSL::Hubbard::Particle, {Nt}));
    std::filesystem::remove(file);
}

// A component whose write was interrupted (no attributes, or 'shape' without the last attribute 'type', and wrong
// values; seen on JURECA when the disk quota was hit) is measured again by a rerun of measure(), not kept.
TEST_CASE("FermionPropagator: an incomplete component is measured again", "[Measurements,propagator]") {
    const NSL::size_t Nt = 8, Nx = 4, Nsrc = 2;
    NSL::Lattice::Ring<Type> lattice(Nx);
    NSL::Parameter params;
    params["name"] = std::string("ring");
    params["beta"] = 2.;
    params["Nt"] = Nt;
    params["Nx"] = Nx;
    params["mu"] = 0.;
    params["device"] = NSL::Device("CPU");
    params["overwrite"] = false;
    params["Number Time Sources"] = Nsrc;

    const std::string file = "test_fermionPropagator_incomplete.h5";
    std::filesystem::remove(file);
    NSL::H5IO h5(file, NSL::File::Truncate);
    NSL::Tensor<Type> phi(Nt, Nx);
    phi.randn();
    phi.imag() = 0;
    h5.write(phi, "ring/markovChain/0/phi");

    using Propagator = NSL::Measure::Hubbard::FermionPropagator<
        Type, decltype(lattice), NSL::FermionMatrix::HubbardExp<Type, decltype(lattice)>>;
    Propagator invM(lattice, params, h5);
    invM.hfm_.stabilityMethod = "QR";
    invM.measure(0);

    const std::string base = "ring/markovChain/0/propagator/spatial/";
    const std::vector<std::string> damaged = {base + "particle/invM[t,4]", base + "hole/invM[0,t]"};
    std::vector<NSL::Tensor<Type>> before;
    NSL::Tensor<Type> stored(Nt, Nx, Nx);
    for (const std::string & node: damaged) {
        h5.read(stored, node);
        before.push_back(NSL::Tensor<Type>(stored, true));
    }
    h5.read(stored, base + "particle/invM[t,0]");
    const NSL::Tensor<Type> untouched(stored, true);

    // replace both by an interrupted write: wrong values; no attributes / only 'shape'
    HighFive::File & f = h5.getFile();
    const std::vector<std::complex<double>> garbage(Nt * Nx * Nx, std::complex<double>(7., -3.));
    for (std::size_t i = 0; i < damaged.size(); ++i) {
        f.unlink(damaged[i]);
        HighFive::DataSet ds = f.createDataSet<std::complex<double>>(damaged[i], HighFive::DataSpace::From(garbage));
        ds.write(garbage);
        if (i == 1) {
            const std::vector<NSL::size_t> shape = {NSL::size_t(Nt), NSL::size_t(Nx), NSL::size_t(Nx)};
            ds.createAttribute<int>("shape", HighFive::DataSpace::From(shape)).write(shape);
        }
    }
    f.flush();
    REQUIRE_FALSE(NSL::Measure::Hubbard::PropagatorTools::complete(f, damaged[0]));
    REQUIRE_FALSE(NSL::Measure::Hubbard::PropagatorTools::complete(f, damaged[1]));
    REQUIRE(NSL::Measure::Hubbard::PropagatorTools::complete(f, base + "particle/invM[t,0]"));

    invM.measure(0);
    for (std::size_t i = 0; i < damaged.size(); ++i) {
        INFO(damaged[i]);
        CHECK(NSL::Measure::Hubbard::PropagatorTools::complete(f, damaged[i]));
        h5.read(stored, damaged[i]);
        CHECK(maxAbsDiff(stored, before[i]) == 0.);
    }
    h5.read(stored, base + "particle/invM[t,0]");
    CHECK(maxAbsDiff(stored, untouched) == 0.);
    std::filesystem::remove(file);
}
