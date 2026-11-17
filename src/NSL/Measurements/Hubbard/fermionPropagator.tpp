#ifndef NSL_FERMION_PROPAGATOR_TPP
#define NSL_FERMION_PROPAGATOR_TPP

#include "Configuration/Configuration.tpp"
#include "concepts.hpp"
#include "device.tpp"
#include "parameter.tpp"
#include "../measure.hpp"
#include "propagatorTools.tpp"

namespace NSL::Measure::Hubbard {

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
class FermionPropagator: public Measurement {
    public:
        FermionPropagator(LatticeType & lattice, NSL::Parameter params, NSL::H5IO & h5, std::string basenode_):
            Measurement(params, h5),
            hfm_(lattice, params),
            corr_(
                params["device"].template to<NSL::Device>(),
                params["Nt"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>()
            ),
            phi_(
                params["device"].template to<NSL::Device>(),
                params["Nt"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>()
            ),
            basenode_(basenode_),
	expKdiag_(lattice.device(), lattice.sites()),
	Uk_(lattice.device(), lattice.sites(), lattice.sites()),
	Vk_(lattice.device(), lattice.sites(), lattice.sites()),
	PI_U_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    PI_D_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites()),
    PI_V_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    SIGMA_U_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    SIGMA_D_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites()),
    SIGMA_V_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites(),lattice.sites()),
	uu_(lattice.device(), lattice.sites(), lattice.sites()),
	dd_(lattice.device(), lattice.sites()),
	vv_(lattice.device(), lattice.sites(), lattice.sites()),
	vu_(lattice.device(), lattice.sites(), lattice.sites()),
	Qnew_(lattice.device(), lattice.sites(), lattice.sites()),
	Dnew_(lattice.device(), lattice.sites()),
	Vnew_(lattice.device(), lattice.sites(), lattice.sites()),
    Qnewt_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites(), lattice.sites()),
	Dnewt_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites()),
	Vnewt_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites(), lattice.sites()),
    vut_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites(), lattice.sites()),
    invAp1F_Ut_(vut_), // alias of vut_
	invAp1F_Tt_(Qnewt_), // alias of Qnewt_
    Fk_Ut_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    Fk_Dt_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites()),   
	Fk_Vt_(lattice.device(),params["Nt"].template to<NSL::size_t>(),lattice.sites(),lattice.sites()),
	uut_(vut_), // alias of vut_
	ddt_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites()),
	vvt_(lattice.device(), params["Nt"].template to<NSL::size_t>(), lattice.sites(), lattice.sites()),
    Qnewtt_(lattice.device(), params["Nt"].template to<NSL::size_t>()-1, lattice.sites(), lattice.sites()),
	Dnewtt_(lattice.device(), params["Nt"].template to<NSL::size_t>()-1, lattice.sites()),
	Vnewtt_(lattice.device(), params["Nt"].template to<NSL::size_t>()-1, lattice.sites(), lattice.sites())
    {
        PropagatorTools::configureAllocator();
    }

    FermionPropagator(LatticeType & lattice, NSL::Parameter params,NSL::H5IO & h5):
        FermionPropagator(
            lattice,
            params,
            h5, 
            params["name"]
        )
    {}

    /*! Position-space propagators of the free theory (phi = 0) and of all stored
     *  configurations with ID >= startConfig. For each source time s and species
     *  (particle, hole), with <base> = markovChain/<cfgID> or NonInteracting (s = 0 only):
     *    <basenode>/<base>/propagator/spatial/<species>/invM[t,<s>]   column, (Nt, Nx, Nx)
     *    <basenode>/<base>/propagator/spatial/<species>/invM[<s>,t]   row,    (Nt, Nx, Nx)
     *    <basenode>/<base>/propagator/spatial/<species>/invM[t,t]     diagonal (written for s = 0)
     *  Existing components are kept unless params["overwrite"] is true.
     */
    void measure(NSL::size_t startConfig);
    void measure() override { measure(NSL::size_t(0)); }
    //! Position-space propagators of the configuration in phi_ (phi_ = 0: free theory)
    void measure(NSL::size_t NumberTimeSources, NSL::size_t cfgID);

    /*! Momentum-projected propagators, same configurations and components as measure(),
     *  projected on the wall sources w = wallSources_() (each source of params["wallSources"]
     *  followed by its complex conjugate, indices 2k and 2k+1):
     *    <basenode>/<base>/propagator/momentum/<species>/invM[t,<s>]/<kSink>-<kSrc>   (Nt, bands, bands)
     *  and likewise invM[<s>,t] and invM[t,t], for all pairs (kSink, kSrc), with entry
     *  (t, sigmaSink, sigmaSrc) = <w(kSink, sigmaSink) | invM(t) | w(kSrc, sigmaSrc)>; and
     *    <basenode>/<base>/propagator/momentum/<species>/Q = 1/Nt sum_t Tr invM[t,t].
     *  The sources are recorded in <basenode>/Meta/propagator/momentum/wallSources (and
     *  params["momenta"], if given, in .../momenta). Both bases measure the same quantities.
     */
    void measureK(NSL::size_t startConfig = 0);
    //! Momentum-projected propagators of the configuration in phi_ (phi_ = 0: free theory)
    void measureK(NSL::size_t NumberTimeSources, NSL::size_t cfgID);

    /*! Propagators projected on eigenvectors of the hopping matrix, same configurations and
     *  components as measure(). The eigenvectors are selected by params["eigenvalueIndices"]
     *  (energy indices) or else by the Fermi window params["fermiWindowBelow"] (default 1) and
     *  params["fermiWindowAbove"] (default 0); see PropagatorTools::selectHoppingEigenbasis. They
     *  form one block of m bands, written as one dataset per component:
     *    <basenode>/<base>/propagator/eigenbasis/<species>/invM[t,<s>]   (Nt, m, m)
     *  with entry (t, i, j) = <v_i | invM(t) | v_j>, likewise invM[<s>,t] and invM[t,t], and
     *    <basenode>/<base>/propagator/eigenbasis/<species>/Q = 1/Nt sum_t Tr invM[t,t].
     *  The selection and the vectors are recorded in <basenode>/Meta/propagator/eigenbasis and
     *  reused on later runs (PropagatorTools::recordEigenbasis).
     */
    void measureEigenbasis(NSL::size_t startConfig = 0);
    //! Eigenbasis projections of the configuration in phi_ (phi_ = 0: free theory)
    void measureEigenbasis(NSL::size_t NumberTimeSources, NSL::size_t cfgID);
    
    void measureDiagonal();
    void measureRow(NSL::size_t tsrc);
    void measureColumn(NSL::size_t tsrc);

    void calcPiSigma(NSL::size_t tsrc);

    // maybe make these private later. . .
    NSL::Tensor<Type> expKdiag_, Uk_, Vk_;
    NSL::Tensor<Type> PI_U_; //(device,Nt,Nx,Nx); // stores U of M = U.D.V [ = Q.D.(D^{-1}.R) ]
    NSL::Tensor<Type> PI_D_; //(device,Nt,Nx);    // stores D of M = U.D.V
    NSL::Tensor<Type> PI_V_; //(device,Nt,Nx,Nx); // stores V of M = U.D.V
    NSL::Tensor<Type> SIGMA_U_; //(device,Nt,Nx,Nx); // stores U of M = U.D.V [ = Q.D.(D^{-1}.R) ]
    NSL::Tensor<Type> SIGMA_D_; //(device,Nt,Nx);    // stores D of M = U.D.V
    NSL::Tensor<Type> SIGMA_V_; //(device,Nt,Nx,Nx); // stores V of M = U.D.V
    NSL::Tensor<Type> uu_; //(device, Nx, Nx);
    NSL::Tensor<Type> dd_; //(device, Nx);
    NSL::Tensor<Type> vv_; //(device, Nx, Nx);
    NSL::Tensor<Type> vu_; //(device, Nx, Nx);
    NSL::Tensor<Type> Qnew_; //(device, Nx, Nx);
    NSL::Tensor<Type> Dnew_; //(device, Nx);
    NSL::Tensor<Type> Vnew_; //(device, Nx, Nx);
    NSL::Tensor<Type> Qnewt_; //(device, Nt, Nx, Nx);
    NSL::Tensor<Type> Dnewt_; //(device, Nt, Nx);
    NSL::Tensor<Type> Vnewt_; //(device, Nt, Nx, Nx);
    NSL::Tensor<Type> vut_; //(device, Nt, Nx, Nx);
    NSL::Tensor<Type> invAp1F_Ut_;
    NSL::Tensor<Type> invAp1F_Tt_;
    NSL::Tensor<Type> Fk_Ut_; //(device,Nt,Nx,Nx);
    NSL::Tensor<Type> Fk_Dt_; //(device,Nt,Nx);   
    NSL::Tensor<Type> Fk_Vt_; //(device,Nt,Nx,Nx);
    NSL::Tensor<Type> uut_; //(device, Nt, Nx, Nx);
    NSL::Tensor<Type> ddt_; //(device, Nt, Nx);
    NSL::Tensor<Type> vvt_; //(device, Nt, Nx, Nx);
    NSL::Tensor<Type> Qnewtt_; //(device, Nt-1, Nx, Nx);
    NSL::Tensor<Type> Dnewtt_; //(device, Nt-1, Nx);
    NSL::Tensor<Type> Vnewtt_; //(device, Nt-1, Nx, Nx);

    FermionMatrixType hfm_;
    NSL::Hubbard::Species species_;

    protected:
    //! Name of the current species in the HDF5 layout
    std::string speciesName_() const {
        return species_ == NSL::Hubbard::Particle ? "particle" : "hole";
    }

    //! Node, relative to basenode_, of one component: <base>/propagator/<block>/<species>/<component>
    std::string node_(const std::string & block, bool free, NSL::size_t cfgID, const std::string & component) const {
        const std::string base = free ? std::string("/NonInteracting") : fmt::format("/markovChain/{}", cfgID);
        return fmt::format("{}/propagator/{}/{}/{}", base, block, speciesName_(), component);
    }

    //! true if the node exists and is kept; with params["overwrite"] an existing node is
    //! deleted (and false returned) so that it is computed and written again
    bool skip_(const std::string & node){
        const std::string full = std::string(basenode_) + node;
        if (!this->h5_.exist(full)) { return false; }
        if (this->params_["overwrite"].template to<bool>()) {
            this->h5_.deleteData(full);
            return false;
        }
        return true;
    }

    //! The wall sources of the momentum projection: each source of params["wallSources"]
    //! (shape blocks x bands x Nx) followed by its complex conjugate (indices 2k and 2k+1)
    NSL::Tensor<Type> wallSources_(){
        NSL::Tensor<Type> src(params_["wallSources"]);
        NSL::Tensor<Type> w(params_["device"].template to<NSL::Device>(), 2 * src.shape(0), src.shape(1), src.shape(2));
        w(NSL::Slice(0, NSL::none_t(), 2), NSL::Ellipsis()) = src;
        w(NSL::Slice(1, NSL::none_t(), 2), NSL::Ellipsis()) = NSL::LinAlg::conj(src);
        return w;
    }

    //! Writes the projections P = PropagatorTools::projectAll(corr_, w) of one component: for the
    //! momentum block one dataset per pair, node/<kSink>-<kSrc> of shape (Nt, bands, bands); for
    //! the eigenbasis (a single block) one dataset (Nt, m, m) at node. Nothing is written if P
    //! has a non-finite entry.
    void writeProjected_(const torch::Tensor & P, const std::string & block, const std::string & node){
        using torch::indexing::Slice;
        // a SIGTERM waits until all leaves of the component are written (NSL::Stop)
        NSL::Stop::WriteGuard writing;
        PropagatorTools::requireFinite(P, std::string(basenode_) + node);
        if (block == "eigenbasis") {
            NSL::Tensor<Type> out(P.index({Slice(), 0, Slice(), 0, Slice()}).contiguous());
            this->h5_.write(out, std::string(basenode_) + node);
            return;
        }
        for (int64_t kSrc = 0; kSrc < P.size(3); ++kSrc) {
            for (int64_t kSink = 0; kSink < P.size(1); ++kSink) {
                NSL::Tensor<Type> out(P.index({Slice(), kSink, Slice(), kSrc, Slice()}).contiguous());
                this->h5_.write(out, std::string(basenode_) + node + fmt::format("/{}-{}", kSink, kSrc));
            }
        }
    }

    //! Projected propagators (block "momentum" or "eigenbasis") of the configuration in phi_
    void measureProjected_(const std::string & block, NSL::Tensor<Type> & w, NSL::size_t NumberTimeSources, NSL::size_t cfgID);

    //! Selects (or reuses) the hopping eigenvectors and sets eigSources_; runs once
    void setupEigenbasis_(){
        if (eigenbasisReady_) { return; }
        PropagatorTools::EigenbasisRequest req;
        if (this->params_.contains("eigenvalueIndices")) {
            const torch::Tensor idx = static_cast<torch::Tensor>(NSL::Tensor<double>(this->params_["eigenvalueIndices"])).to(torch::kCPU).to(torch::kLong).contiguous();
            req.energyIndices = std::vector<int64_t>(idx.data_ptr<int64_t>(), idx.data_ptr<int64_t>() + idx.numel());
        }
        if (this->params_.contains("fermiWindowBelow")) { req.below = this->params_["fermiWindowBelow"].template to<int>(); }
        if (this->params_.contains("fermiWindowAbove")) { req.above = this->params_["fermiWindowAbove"].template to<int>(); }
        const double mu = this->params_.contains("mu") ? this->params_["mu"].template to<double>() : 0.;
        const torch::Tensor K = static_cast<torch::Tensor>(hfm_.Lat.hopping_matrix());
        const PropagatorTools::EigenbasisSelection sel = PropagatorTools::selectHoppingEigenbasis(K, mu, req);
        const torch::Tensor V = PropagatorTools::recordEigenbasis<Type>(
            this->h5_, std::string(basenode_) + "/Meta/propagator/eigenbasis", sel, this->params_["overwrite"].template to<bool>());
        // one block whose m bands are the selected vectors: (1, m, Nx)
        const NSL::Device device = this->params_["device"].template to<NSL::Device>();
        eigSources_ = NSL::Tensor<Type>(V.transpose(0, 1).unsqueeze(0).contiguous().to(device.device().device()));
        eigenbasisReady_ = true;
    }

    //! The selected hopping eigenvectors as the m bands of one block, (1, m, Nx)
    NSL::Tensor<Type> eigSources_;
    bool eigenbasisReady_ = false;

    //! Writes value to node (relative to basenode_) or checks that the stored value is the
    //! same; a different stored value is replaced with params["overwrite"], otherwise an error
    template<NSL::Concept::isNumber T>
    void recordOrCheck_(NSL::Tensor<T> & value, const std::string & node){
        const std::string full = std::string(basenode_) + node;
        if (this->h5_.exist(full)) {
            NSL::Tensor<T> stored(value, true);
            this->h5_.read(stored, full);
            const torch::Tensor a = static_cast<torch::Tensor>(stored).cpu();
            const torch::Tensor b = static_cast<torch::Tensor>(value).cpu();
            if (a.sizes() == b.sizes() && torch::equal(a, b)) { return; }
            if (!this->params_["overwrite"].template to<bool>()) {
                throw std::runtime_error(fmt::format(
                    "{} differs from the value in the parameter file, so new momentum data would not match "
                    "the stored data. Use the same wall sources and momenta, another file, or --overwrite "
                    "to measure all momentum data again.", full));
            }
            this->h5_.deleteData(full);
        }
        this->h5_.write(value, full);
    }

    //! Records the wall sources used (and the momentum labels, if given) in Meta/propagator/momentum
    void writeMomentumMeta_(NSL::Tensor<Type> & w){
        recordOrCheck_(w, "/Meta/propagator/momentum/wallSources");
        if (this->params_.contains("momenta")) {
            NSL::Tensor<double> momenta = this->params_["momenta"];
            recordOrCheck_(momenta, "/Meta/propagator/momentum/momenta");
        }
    }

    //! Reads every stored configuration with ID >= startConfig into phi_ and calls f(cfgID)
    template<typename Function>
    void forConfigurations_(NSL::size_t startConfig, Function f){
        auto [minCfg, maxCfg] = this->h5_.getMinMaxConfigs(std::string(basenode_) + "/markovChain");
        const NSL::size_t firstCfg = std::max<NSL::size_t>(minCfg, startConfig);
        NSL::Logger::info("Found configurations {} to {}; measuring from {}", minCfg, maxCfg, firstCfg);
        for (NSL::size_t cfgID = firstCfg; cfgID <= NSL::size_t(maxCfg); ++cfgID) {
            // configuration IDs need not be contiguous; a missing field would otherwise leave the
            // previous phi_ in place and write its propagators under this ID
            const std::string phiNode = fmt::format("{}/markovChain/{}/phi", std::string(basenode_), cfgID);
            if (!this->h5_.exist(phiNode)) { continue; }
            // safe early stop (NSL::Stop): every component written so far is complete
            if (NSL::Stop::requested()) {
                NSL::Logger::warn("Measurement stopped ({}) before configuration {}. Run again to continue.",
                    NSL::Stop::signalName(), cfgID);
                break;
            }
            NSL::Logger::info("Measuring configuration {}/{}", cfgID, maxCfg);
            this->h5_.read(phi_, phiNode);
            f(cfgID);
        }
    }

    NSL::Tensor<Type> corr_;
    NSL::Tensor<Type> phi_;

    std::string basenode_;
};

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measure(NSL::size_t NumberTimeSources, NSL::size_t cfgID){
    const bool free = (phi_ == Type(0)).all();
    const NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
    const NSL::size_t tsrcStep = ceil((Nt+0.0)/NumberTimeSources);
    NSL::size_t kept = 0;

    corr_ = Type(0);
    for (const NSL::Hubbard::Species species: {NSL::Hubbard::Particle, NSL::Hubbard::Hole}) {
        species_ = species;
        for (NSL::size_t tsrc = 0; tsrc < Nt; tsrc += tsrcStep) {
            // the fermion matrix and the prefix/suffix products of this source time are
            // computed once, and only if a component has to be measured
            bool ready = false;
            auto component = [&](const std::string & name, auto && compute){
                const std::string node = node_("spatial", free, cfgID, name);
                if (skip_(node)) { ++kept; return; }
                if (NSL::Stop::requested()) { return; }
                if (!ready) {
                    hfm_.populate(phi_, species_);
                    calcPiSigma(tsrc);
                    ready = true;
                }
                compute();
                PropagatorTools::requireFinite(corr_, std::string(basenode_) + node);
                this->h5_.write(corr_, std::string(basenode_) + node);
            };
            component(fmt::format("invM[t,{}]", tsrc), [&](){ measureColumn(tsrc); });
            component(fmt::format("invM[{},t]", tsrc), [&](){ measureRow(tsrc); });
            // the diagonal uses the prefix/suffix products of source time 0
            if (tsrc == 0) { component("invM[t,t]", [&](){ measureDiagonal(); }); }
        }
    }
    if (kept > 0) {
        NSL::Logger::info("{}: kept {} existing spatial component(s)", free ? std::string("free theory") : fmt::format("configuration {}", cfgID), kept);
    }
} // measure(Ntsrc, cfgID)

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measure(NSL::size_t startConfig){
    NSL::Logger::info("Start measuring position-space propagators (Hubbard::FermionPropagator)");

    // free theory: phi = 0, source time 0
    phi_ = Type(0);
    measure(1, 0);

    const NSL::size_t NumberTimeSources = this->params_["Number Time Sources"].template to<NSL::size_t>();
    forConfigurations_(startConfig, [&](NSL::size_t cfgID){ measure(NumberTimeSources, cfgID); });
} // measure(startConfig)

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measureProjected_(const std::string & block, NSL::Tensor<Type> & w, NSL::size_t NumberTimeSources, NSL::size_t cfgID){
    const bool free = (phi_ == Type(0)).all();
    const NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
    const NSL::size_t tsrcStep = ceil((Nt+0.0)/NumberTimeSources);
    NSL::size_t kept = 0;

    corr_ = Type(0);
    for (const NSL::Hubbard::Species species: {NSL::Hubbard::Particle, NSL::Hubbard::Hole}) {
        species_ = species;
        for (NSL::size_t tsrc = 0; tsrc < Nt; tsrc += tsrcStep) {
            // the fermion matrix and the prefix/suffix products of this source time are
            // computed once, and only if a component has to be measured
            bool ready = false;
            auto prepare = [&](){
                if (ready) { return; }
                hfm_.populate(phi_, species_);
                calcPiSigma(tsrc);
                ready = true;
            };
            auto component = [&](const std::string & name, auto && compute){
                const std::string node = node_(block, free, cfgID, name);
                if (skip_(node)) { ++kept; return; }
                if (NSL::Stop::requested()) { return; }
                prepare();
                compute();
                writeProjected_(PropagatorTools::projectAll(corr_, w), block, node);
            };
            component(fmt::format("invM[t,{}]", tsrc), [&](){ measureColumn(tsrc); });
            component(fmt::format("invM[{},t]", tsrc), [&](){ measureRow(tsrc); });

            // the diagonal, and Q = 1/Nt sum_t Tr invM[t,t], use the products of source time 0
            if (tsrc == 0) {
                const std::string nodeD = node_(block, free, cfgID, "invM[t,t]");
                const std::string nodeQ = node_(block, free, cfgID, "Q");
                const bool needD = !skip_(nodeD), needQ = !skip_(nodeQ);
                kept += !needD + !needQ;
                if ((needD || needQ) && !NSL::Stop::requested()) {
                    prepare();
                    measureDiagonal();
                    if (needD) { writeProjected_(PropagatorTools::projectAll(corr_, w), block, nodeD); }
                    if (needQ) {
                        Type Q = 0;
                        for (NSL::size_t t = 0; t < Nt; ++t) {
                            Q += NSL::LinAlg::diag(corr_(t, NSL::Ellipsis())).sum();
                        }
                        Q /= Type(Nt);
                        PropagatorTools::requireFiniteScalar(Q, std::string(basenode_) + nodeQ);
                        this->h5_.write(Q, std::string(basenode_) + nodeQ);
                    }
                }
            }
        }
    }
    if (kept > 0) {
        NSL::Logger::info("{}: kept {} existing {} component(s)", free ? std::string("free theory") : fmt::format("configuration {}", cfgID), kept, block);
    }
} // measureProjected_(block, w, Ntsrc, cfgID)

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measureK(NSL::size_t NumberTimeSources, NSL::size_t cfgID){
    NSL::Tensor<Type> w = wallSources_();
    measureProjected_("momentum", w, NumberTimeSources, cfgID);
} // measureK(Ntsrc, cfgID)

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measureEigenbasis(NSL::size_t NumberTimeSources, NSL::size_t cfgID){
    setupEigenbasis_();
    measureProjected_("eigenbasis", eigSources_, NumberTimeSources, cfgID);
} // measureEigenbasis(Ntsrc, cfgID)

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measureEigenbasis(NSL::size_t startConfig){
    NSL::Logger::info("Start measuring hopping-eigenbasis projected propagators (Hubbard::FermionPropagator)");
    setupEigenbasis_();

    // free theory: phi = 0, source time 0
    phi_ = Type(0);
    measureEigenbasis(1, 0);

    const NSL::size_t NumberTimeSources = this->params_["Number Time Sources"].template to<NSL::size_t>();
    forConfigurations_(startConfig, [&](NSL::size_t cfgID){ measureEigenbasis(NumberTimeSources, cfgID); });
} // measureEigenbasis(startConfig)

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measureK(NSL::size_t startConfig){
    NSL::Logger::info("Start measuring momentum-projected propagators (Hubbard::FermionPropagator)");

    NSL::Tensor<Type> w = wallSources_();
    writeMomentumMeta_(w);

    // free theory: phi = 0, source time 0
    phi_ = Type(0);
    measureK(1, 0);

    const NSL::size_t NumberTimeSources = this->params_["Number Time Sources"].template to<NSL::size_t>();
    forConfigurations_(startConfig, [&](NSL::size_t cfgID){ measureK(NumberTimeSources, cfgID); });
} // measureK(startConfig)


template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measureDiagonal(){
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
    NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();

    // t=0 term
    { auto s = NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint_view(SIGMA_U_(0,NSL::Ellipsis())), false);
      NSL::LinAlg::add_diagonal(s, SIGMA_D_(0,NSL::Ellipsis()));
      std::tie( Qnew_, Dnew_, Vnew_ ) = PropagatorTools::udt( s ); }

    invAp1F_Ut_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnew_));
    invAp1F_Tt_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint_view(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));

    corr_(0,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(invAp1F_Ut_(0,NSL::Ellipsis()),invAp1F_Tt_(0,NSL::Ellipsis()));

    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_V_(NSL::Slice(0,Nt-1),NSL::Ellipsis()),SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()));
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) *= PI_D_(NSL::Slice(0,Nt-1),NSL::Ellipsis()).expand_view(Nx,2);
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) *= SIGMA_D_(NSL::Slice(1,Nt),NSL::Ellipsis()).expand_view(Nx,1);

    if (!this->hfm_.stabilityMethod.compare("SVD")) {
	   std::tie( uut_,ddt_,vvt_ ) = NSL::LinAlg::svd(vut_);
	} else { //(!this->hfm_.stabilityMethod.compare("QR")) {
	   std::tie( uut_,ddt_,vvt_ ) = PropagatorTools::udt(vut_);
	}

    Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()),uut_(NSL::Slice(1,Nt),NSL::Ellipsis()));
    Fk_Dt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = ddt_(NSL::Slice(1,Nt),NSL::Ellipsis());
	Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vvt_(NSL::Slice(1,Nt),NSL::Ellipsis()),SIGMA_V_(NSL::Slice(1,Nt),NSL::Ellipsis()));
    { auto s = NSL::LinAlg::solve(Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis()), NSL::LinAlg::adjoint_view(Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis())), false);
      NSL::LinAlg::add_diagonal(s, Fk_Dt_(NSL::Slice(1,Nt),NSL::Ellipsis()));
      std::tie( Qnewtt_, Dnewtt_, Vnewtt_ ) = PropagatorTools::udt( s ); }
    invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnewtt_, Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnewtt_));
    invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::adjoint_view(NSL::LinAlg::mat_mul( Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()), Qnewtt_ ));
    
	corr_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()),invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()));
} // measureDiagonal()

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measureRow(NSL::size_t tsrc){
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();

    // t=0 term
    { auto s = NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint_view(SIGMA_U_(0,NSL::Ellipsis())), false);
      NSL::LinAlg::add_diagonal(s, SIGMA_D_(0,NSL::Ellipsis()));
      std::tie( Qnew_, Dnew_, Vnew_ ) = PropagatorTools::udt( s ); }

    invAp1F_Ut_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnew_));
    invAp1F_Tt_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint_view(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));

    corr_(tsrc,NSL::Ellipsis()) = NSL::LinAlg::mat_mul( invAp1F_Ut_(0,NSL::Ellipsis()),invAp1F_Tt_(0,NSL::Ellipsis()) );

    // t>0 terms
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::adjoint(PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),NSL::LinAlg::solve(SIGM\
A_V_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(1./SIGMA_D_(NSL::Slice(1,Nt),NSL::Ellipsis()))));
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) + NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(NSL::LinAlg::diag_embed(P\
I_D_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),PI_V_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()));

    if (!this->hfm_.stabilityMethod.compare("SVD")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::svd(vut_);
    } else { // (!this->hfm_.stabilityMethod.compare("QR")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = PropagatorTools::udt(vut_);
    }

    invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::solve(Vnewt_(NSL::Slice(\
1,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(1,Nt),NSL::Ellipsis()))));
    invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()), Qnewt_(NSL::Sli\
ce(1,Nt),NSL::Ellipsis())));

    for (int t=1;t<Nt;t++){ // t>0 terms

      if((t+tsrc)%Nt > tsrc ) {
        corr_((t+tsrc)%Nt,NSL::Ellipsis()) = -NSL::LinAlg::mat_mul(invAp1F_Ut_(t,NSL::Ellipsis()),invAp1F_Tt_(t,NSL::Ellipsis()));
      } else {
        corr_((t+tsrc)%Nt,NSL::Ellipsis()) =  NSL::LinAlg::mat_mul(invAp1F_Ut_(t,NSL::Ellipsis()),invAp1F_Tt_(t,NSL::Ellipsis()));
      }

    }
    // do something and store in corr_

} // measureRow(tsrc)

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::measureColumn(NSL::size_t tsrc){
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();


    // t=0 term
    { auto s = NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint_view(SIGMA_U_(0,NSL::Ellipsis())), false);
      NSL::LinAlg::add_diagonal(s, SIGMA_D_(0,NSL::Ellipsis()));
      std::tie( Qnew_, Dnew_, Vnew_ ) = PropagatorTools::udt( s ); }

    invAp1F_Ut_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnew_));
    invAp1F_Tt_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint_view(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));

    corr_(tsrc,NSL::Ellipsis()) = NSL::LinAlg::mat_mul( invAp1F_Ut_(0,NSL::Ellipsis()),invAp1F_Tt_(0,NSL::Ellipsis()) );

    // t>0
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::solve(PI_V_(NSL::Slice(0,Nt-1),NSL::Ellipsis()), NSL::LinAlg::adjoint(\
SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis())), false ),NSL::LinAlg::diag_embed(1./PI_D_(NSL::Slice(0,Nt-1),NSL::Ellipsis())));
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = vut_(NSL::Slice(1,Nt),NSL::Ellipsis())+ NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(NSL::LinAlg::diag_embe\
d(SIGMA_D_(NSL::Slice(1,Nt),NSL::Ellipsis())),SIGMA_V_(NSL::Slice(1,Nt),NSL::Ellipsis())),PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()));

   if (!this->hfm_.stabilityMethod.compare("SVD")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::svd(vut_); 
    } else { // (!this->hfm_.stabilityMethod.compare("QR")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = PropagatorTools::udt(vut_);
    }


    invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()),NSL::LinAlg::solve(Vnewt_(NSL::Sli\
ce(1,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(1,Nt),NSL::Ellipsis()))));
    invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()), Qnewt_(NSL\
::Slice(1,Nt),NSL::Ellipsis())));

    for (int t=1;t<Nt;t++){ // t>0 terms

        if((t+tsrc)%Nt > tsrc){
            corr_((t+tsrc)%Nt,NSL::Ellipsis()) =  NSL::LinAlg::mat_mul(invAp1F_Ut_(t,NSL::Ellipsis()),invAp1F_Tt_(t,NSL::Ellipsis()));
        } else {
            corr_((t+tsrc)%Nt,NSL::Ellipsis()) = -NSL::LinAlg::mat_mul(invAp1F_Ut_(t,NSL::Ellipsis()),invAp1F_Tt_(t,NSL::Ellipsis()));
        }
     
    }


} // measureColumn(tsrc)

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagator<Type,LatticeType,FermionMatrixType>::calcPiSigma(NSL::size_t tsrc){

     NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
     NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();

    // calculation of F_k(t) (= f^{-1}_k(t)) using initial SVD
    std::tie(Uk_, expKdiag_, Vk_) = this->hfm_.Lat.svd_hopping(hfm_.sgn_* hfm_.delta_);  // note Uk.expKdiag.Vk = expK
    PI_V_ = Vk_ * NSL::LinAlg::shift(this->hfm_.phiExp_,-tsrc).expand(Nx).transpose(1,2);
    PI_D_ = expKdiag_*NSL::LinAlg::exp(hfm_.sgn_*hfm_.mu_);
    PI_U_ = Uk_;

    SIGMA_V_=PI_V_;
    SIGMA_D_=PI_D_;
    SIGMA_U_=PI_U_;

    if (!this->hfm_.stabilityMethod.compare("SVD")) {
        for (int t=1;t<Nt;t++) {
      	  // PI(t)=Fk(t)...Fk(0)
      	  vu_ = NSL::LinAlg::mat_mul(PI_V_(t,NSL::Ellipsis()),PI_U_(t-1,NSL::Ellipsis()));
      	  vu_ *= PI_D_(t,NSL::Ellipsis()).expand_view(Nx, 1);   // O(Nx^2) left-scale
      	  vu_ *= PI_D_(t-1,NSL::Ellipsis()).expand_view(Nx, 0); // O(Nx^2) right-scale
      	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::svd(vu_);
      	  PI_U_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(t,NSL::Ellipsis()),uu_);
        PI_D_(t,NSL::Ellipsis()) = dd_;
        PI_V_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,PI_V_(t-1,NSL::Ellipsis()));

        // SIGMA(t)=Fk(Nt-1)...Fk(t)
        vu_ = NSL::LinAlg::mat_mul(SIGMA_V_(Nt-t,NSL::Ellipsis()),SIGMA_U_(Nt-1-t,NSL::Ellipsis()));
        vu_ *= SIGMA_D_(Nt-1-t,NSL::Ellipsis()).expand_view(Nx, 0); // O(Nx^2) right-scale
        vu_ *= SIGMA_D_(Nt-t,NSL::Ellipsis()).expand_view(Nx, 1);   // O(Nx^2) left-scale
        std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::svd(vu_);
        SIGMA_U_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_U_(Nt-t,NSL::Ellipsis()),uu_);
        SIGMA_D_(Nt-1-t,NSL::Ellipsis()) = dd_;
        SIGMA_V_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,SIGMA_V_(Nt-1-t,NSL::Ellipsis()));
        }
    } else { //(!this->hfm_.stabilityMethod.compare("QR")) {
       for (int t=1;t<Nt;t++) {
      	  // PI(t)=Fk(t)...Fk(0)
      	  vu_ = NSL::LinAlg::mat_mul(PI_V_(t,NSL::Ellipsis()),PI_U_(t-1,NSL::Ellipsis()));
      	  vu_ *= PI_D_(t,NSL::Ellipsis()).expand_view(Nx, 1);   // O(Nx^2) left-scale
      	  vu_ *= PI_D_(t-1,NSL::Ellipsis()).expand_view(Nx, 0); // O(Nx^2) right-scale
      	  std::tie( uu_,dd_,vv_ ) = PropagatorTools::udt(vu_);
      	  PI_U_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(t,NSL::Ellipsis()),uu_);
	  PI_D_(t,NSL::Ellipsis()) = dd_;
	  PI_V_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,PI_V_(t-1,NSL::Ellipsis()));

	  // SIGMA(t)=Fk(Nt-1)...Fk(t)
	  vu_ = NSL::LinAlg::mat_mul(SIGMA_V_(Nt-t,NSL::Ellipsis()),SIGMA_U_(Nt-1-t,NSL::Ellipsis()));
	  vu_ *= SIGMA_D_(Nt-1-t,NSL::Ellipsis()).expand_view(Nx, 0); // O(Nx^2) right-scale
	  vu_ *= SIGMA_D_(Nt-t,NSL::Ellipsis()).expand_view(Nx, 1);   // O(Nx^2) left-scale
	  std::tie( uu_,dd_,vv_ ) = PropagatorTools::udt(vu_);
	  SIGMA_U_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_U_(Nt-t,NSL::Ellipsis()),uu_);
	  SIGMA_D_(Nt-1-t,NSL::Ellipsis()) = dd_;
	  SIGMA_V_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,SIGMA_V_(Nt-1-t,NSL::Ellipsis()));
      }
    }

} // calcPiSigma(tsrc)


} // namespace NSL::Measure::Hubbard

#endif // NSL_FERMION_PROPAGATOR_TPP

