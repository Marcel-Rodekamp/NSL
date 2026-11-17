#ifndef NSL_FERMION_PROPAGATOR_SPIN_BASIS_TPP
#define NSL_FERMION_PROPAGATOR_SPIN_BASIS_TPP

#include "Configuration/Configuration.tpp"
#include "concepts.hpp"
#include "device.tpp"
#include "parameter.tpp"
#include "../measure.hpp"

namespace NSL::Measure::Hubbard {

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
class FermionPropagatorSpinBasis: public Measurement {
    public:
        FermionPropagatorSpinBasis(LatticeType & lattice, NSL::Parameter params, NSL::H5IO & h5, std::string basenode_):
            Measurement(params, h5),
            hfm_(lattice, params),
            corr_(
                params["device"].template to<NSL::Device>(),
                params["Nt"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>()
            ),
            // momentum projected correlators (measureK) need wall sources; position
            // space measurements (measure) do not, then a 1-band placeholder is used
            corrK_(
                params["device"].template to<NSL::Device>(),
                params["Nt"].template to<NSL::size_t>(),
                params.contains("wallSources") ? params["wallSources"].shape(1).template to<NSL::size_t>() : NSL::size_t(1), // bands
                params.contains("wallSources") ? params["wallSources"].shape(1).template to<NSL::size_t>() : NSL::size_t(1)  // bands
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
    {}

    FermionPropagatorSpinBasis(LatticeType & lattice, NSL::Parameter params,NSL::H5IO & h5):
        FermionPropagatorSpinBasis(
            lattice,
            params,
            h5, 
            params["name"]
        )
    {}

    /*! Position-space propagators of the free theory (phi = 0) and of all stored
     *  configurations with ID >= startConfig. For each source time s and spin
     *  (spinUp, spinDown), with <base> = markovChain/<cfgID> or NonInteracting (s = 0 only):
     *    <basenode>/<base>/propagator/spatial/<spin>/invM[t,<s>]   column, (Nt, Nx, Nx)
     *    <basenode>/<base>/propagator/spatial/<spin>/invM[<s>,t]   row,    (Nt, Nx, Nx)
     *    <basenode>/<base>/propagator/spatial/<spin>/invM[t,t]     diagonal (written for s = 0)
     *  Existing components are kept unless params["overwrite"] is true.
     */
    void measure(NSL::size_t startConfig);
    void measure() override { measure(NSL::size_t(0)); }
    //! Position-space propagators of the configuration in phi_ (phi_ = 0: free theory)
    void measure(NSL::size_t NumberTimeSources, NSL::size_t cfgID);

    /*! Momentum-projected propagators, same configurations as measure(): the columns
     *  projected on the wall sources w = params["wallSources"] with equal sink and source
     *  momentum,
     *    <basenode>/<base>/propagator/momentum/<spin>/invM[t,<s>]/<k>-<k>   (Nt, bands, bands)
     *  and Q = 1/Nt sum_t Tr invM[t,t] in <basenode>/<base>/propagator/momentum/<spin>/Q.
     *  The sources are recorded in <basenode>/Meta/propagator/momentum/wallSources (and
     *  params["momenta"], if given, in .../momenta).
     */
    void measureK(NSL::size_t startConfig = 0);
    //! Momentum-projected propagators of the configuration in phi_ (phi_ = 0: free theory)
    void measureK(NSL::size_t NumberTimeSources, NSL::size_t cfgID);
    
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
    NSL::Tensor<Type> Qnewt_; //(device, Nx, Nx);
    NSL::Tensor<Type> Dnewt_; //(device, Nx);
    NSL::Tensor<Type> Vnewt_; //(device, Nx, Nx);
    NSL::Tensor<Type> vut_; //(device, Nx, Nx);
    NSL::Tensor<Type> invAp1F_Ut_;
    NSL::Tensor<Type> invAp1F_Tt_;
    NSL::Tensor<Type> Fk_Ut_; //(device,Nx,Nx);
    NSL::Tensor<Type> Fk_Dt_; //(device,Nx);   
    NSL::Tensor<Type> Fk_Vt_; //(device,Nx,Nx);
    NSL::Tensor<Type> uut_; //(device, Nx, Nx);
    NSL::Tensor<Type> ddt_; //(device, Nx);
    NSL::Tensor<Type> vvt_; //(device, Nx, Nx);
    NSL::Tensor<Type> Qnewtt_; //(device, Nx, Nx);
    NSL::Tensor<Type> Dnewtt_; //(device, Nx);
    NSL::Tensor<Type> Vnewtt_; //(device, Nx, Nx);

    FermionMatrixType hfm_;
    NSL::Hubbard::Spin spin_;

    protected:
    //! Name of the current spin in the HDF5 layout
    std::string speciesName_() const {
        return spin_ == NSL::Hubbard::Up ? "spinUp" : "spinDown";
    }

    //! Node, relative to basenode_, of one component: <base>/propagator/<block>/<spin>/<component>
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

    //! The wall sources of the momentum projection, params["wallSources"] (blocks x bands x Nx)
    NSL::Tensor<Type> wallSources_(){
        NSL::Tensor<Type> src(params_["wallSources"]);
        NSL::Tensor<Type> w(params_["device"].template to<NSL::Device>(), src.shape(0), src.shape(1), src.shape(2));
        w(NSL::Ellipsis()) = src;
        return w;
    }

    //! corrK_(t, sigmaSink, sigmaSrc) = <w(kSink, sigmaSink) | corr_(t) | w(kSrc, sigmaSrc)>
    void project_(NSL::Tensor<Type> & w, NSL::size_t kSink, NSL::size_t kSrc){
        const NSL::size_t bDim = w.shape(1);
        for (NSL::size_t sigmaSink = 0; sigmaSink < bDim; ++sigmaSink) {
            for (NSL::size_t sigmaSrc = 0; sigmaSrc < bDim; ++sigmaSrc) {
                corrK_(NSL::Slice(), sigmaSink, sigmaSrc) = NSL::LinAlg::inner_product(
                    w(kSink, sigmaSink, NSL::Slice()), NSL::LinAlg::mat_mul(corr_, w(kSrc, sigmaSrc, NSL::Slice())), 1
                );
            }
        }
    }

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
            NSL::Logger::info("Measuring configuration {}/{}", cfgID, maxCfg);
            this->h5_.read(phi_, fmt::format("{}/markovChain/{}/phi", std::string(basenode_), cfgID));
            f(cfgID);
        }
    }

    NSL::Tensor<Type> corr_;
    NSL::Tensor<Type> phi_;
    NSL::Tensor<Type> corrK_;

    std::string basenode_;
};

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measure(NSL::size_t NumberTimeSources, NSL::size_t cfgID){
    const bool free = (phi_ == Type(0)).all();
    const NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
    const NSL::size_t tsrcStep = ceil((Nt+0.0)/NumberTimeSources);
    NSL::size_t kept = 0;

    corr_ = Type(0);
    for (const NSL::Hubbard::Spin spin: {NSL::Hubbard::Up, NSL::Hubbard::Down}) {
        spin_ = spin;
        for (NSL::size_t tsrc = 0; tsrc < Nt; tsrc += tsrcStep) {
            // the fermion matrix and the prefix/suffix products of this source time are
            // computed once, and only if a component has to be measured
            bool ready = false;
            auto component = [&](const std::string & name, auto && compute){
                const std::string node = node_("spatial", free, cfgID, name);
                if (skip_(node)) { ++kept; return; }
                if (!ready) {
                    hfm_.populate(phi_, spin_);
                    calcPiSigma(tsrc);
                    ready = true;
                }
                compute();
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
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measure(NSL::size_t startConfig){
    NSL::Logger::info("Start measuring position-space propagators (Hubbard::FermionPropagatorSpinBasis)");

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
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureK(NSL::size_t NumberTimeSources, NSL::size_t cfgID){
    const bool free = (phi_ == Type(0)).all();
    const NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
    const NSL::size_t tsrcStep = ceil((Nt+0.0)/NumberTimeSources);
    NSL::Tensor<Type> w = wallSources_();
    const NSL::size_t kDim = w.shape(0);
    NSL::size_t kept = 0;

    corr_ = Type(0);
    corrK_ = Type(0);
    for (const NSL::Hubbard::Spin spin: {NSL::Hubbard::Up, NSL::Hubbard::Down}) {
        spin_ = spin;
        for (NSL::size_t tsrc = 0; tsrc < Nt; tsrc += tsrcStep) {
            bool ready = false;
            auto prepare = [&](){
                if (ready) { return; }
                hfm_.populate(phi_, spin_);
                calcPiSigma(tsrc);
                ready = true;
            };

            // columns, projected with equal sink and source momentum
            std::string node = node_("momentum", free, cfgID, fmt::format("invM[t,{}]", tsrc));
            if (skip_(node)) {
                ++kept;
            } else {
                prepare();
                measureColumn(tsrc);
                for (NSL::size_t k = 0; k < kDim; ++k) {
                    project_(w, k, k);
                    this->h5_.write(corrK_, std::string(basenode_) + node + fmt::format("/{}-{}", k, k));
                }
            }

            // Q = 1/Nt sum_t Tr invM[t,t], from the diagonal of source time 0
            if (tsrc == 0) {
                node = node_("momentum", free, cfgID, "Q");
                if (skip_(node)) {
                    ++kept;
                } else {
                    prepare();
                    measureDiagonal();
                    auto Q = phi_[0]*0.0;
                    for (int t=0; t< Nt; t++) {
                        Q += NSL::LinAlg::diag(corr_(t,NSL::Ellipsis())).sum();
                    }
                    Q /= Nt;
                    this->h5_.write(Q, std::string(basenode_) + node);
                }
            }
        }
    }
    if (kept > 0) {
        NSL::Logger::info("{}: kept {} existing momentum component(s)", free ? std::string("free theory") : fmt::format("configuration {}", cfgID), kept);
    }
} // measureK(Ntsrc, cfgID)

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureK(NSL::size_t startConfig){
    NSL::Logger::info("Start measuring momentum-projected propagators (Hubbard::FermionPropagatorSpinBasis)");

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
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureDiagonal(){
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();

    // t=0 term
    std::tie( Qnew_, Dnew_, Vnew_ ) = NSL::LinAlg::udt( NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(0,NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(SIGMA_D_(0,NSL::Ellipsis())));

    invAp1F_Ut_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnew_));
    invAp1F_Tt_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));

    corr_(0,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(invAp1F_Ut_(0,NSL::Ellipsis()),invAp1F_Tt_(0,NSL::Ellipsis()));



    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_V_(NSL::Slice(0,Nt-1),NSL::Ellipsis()),SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()));
	vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::diag_embed(PI_D_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),vut_(NSL::Slice(1,Nt),NSL::Ellipsis()));
	vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(SIGMA_D_(NSL::Slice(1,Nt),NSL::Ellipsis())));


    if (!this->hfm_.stabilityMethod.compare("SVD")) {
	   std::tie( uut_,ddt_,vvt_ ) = NSL::LinAlg::svd(vut_);
	} else { //(!this->hfm_.stabilityMethod.compare("QR")) {
	   std::tie( uut_,ddt_,vvt_ ) = NSL::LinAlg::udt(vut_);
	}


    Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()),uut_(NSL::Slice(1,Nt),NSL::Ellipsis()));
    Fk_Dt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = ddt_(NSL::Slice(1,Nt),NSL::Ellipsis());
	Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vvt_(NSL::Slice(1,Nt),NSL::Ellipsis()),SIGMA_V_(NSL::Slice(1,Nt),NSL::Ellipsis()));

    std::tie( Qnewtt_, Dnewtt_, Vnewtt_ ) = NSL::LinAlg::udt( NSL::LinAlg::solve(Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis()), NSL::LinAlg::adjoint(Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(Fk_Dt_(NSL::Slice(1,Nt),NSL::Ellipsis())));

    invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnewtt_, Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnewtt_));
    invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()), Qnewtt_ ));

	corr_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()),invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()));


} // measureDiagonal()

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureRow(NSL::size_t tsrc){
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();


    // t=0 term
    std::tie( Qnew_, Dnew_, Vnew_ ) = NSL::LinAlg::udt( NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(0,NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(SIGMA_D_(0,NSL::Ellipsis())));

    invAp1F_Ut_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnew_));
    invAp1F_Tt_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));

    corr_(tsrc,NSL::Ellipsis()) = NSL::LinAlg::mat_mul( invAp1F_Ut_(0,NSL::Ellipsis()),invAp1F_Tt_(0,NSL::Ellipsis()) );

    // t>0 terms
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::adjoint(PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),NSL::LinAlg::solve(SIGMA_V_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(1./SIGMA_D_(NSL::Slice(1,Nt),NSL::Ellipsis()))));
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) + NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(NSL::LinAlg::diag_embed(PI_D_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),PI_V_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()));

    if (!this->hfm_.stabilityMethod.compare("SVD")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::svd(vut_);
    } else { //(!this->hfm_.stabilityMethod.compare("QR")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::udt(vut_);
    }

    invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::solve(Vnewt_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(1,Nt),NSL::Ellipsis()))));
    invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()), Qnewt_(NSL::Slice(1,Nt),NSL::Ellipsis())));

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
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureColumn(NSL::size_t tsrc){
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();


    // t=0 term
    std::tie( Qnew_, Dnew_, Vnew_ ) = NSL::LinAlg::udt( NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(0,NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(SIGMA_D_(0,NSL::Ellipsis())));

    invAp1F_Ut_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnew_));
    invAp1F_Tt_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));

    corr_(tsrc,NSL::Ellipsis()) = NSL::LinAlg::mat_mul( invAp1F_Ut_(0,NSL::Ellipsis()),invAp1F_Tt_(0,NSL::Ellipsis()) );

    // t>0
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::solve(PI_V_(NSL::Slice(0,Nt-1),NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis())), false ),NSL::LinAlg::diag_embed(1./PI_D_(NSL::Slice(0,Nt-1),NSL::Ellipsis())));
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = vut_(NSL::Slice(1,Nt),NSL::Ellipsis())+ NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(NSL::LinAlg::diag_embed(SIGMA_D_(NSL::Slice(1,Nt),NSL::Ellipsis())),SIGMA_V_(NSL::Slice(1,Nt),NSL::Ellipsis())),PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()));

    if (!this->hfm_.stabilityMethod.compare("SVD")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::svd(vut_);
    } else { //(!this->hfm_.stabilityMethod.compare("QR")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::udt(vut_);
    }


    invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()),NSL::LinAlg::solve(Vnewt_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(1,Nt),NSL::Ellipsis()))));
    invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()), Qnewt_(NSL::Slice(1,Nt),NSL::Ellipsis())));

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
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::calcPiSigma(NSL::size_t tsrc){

     NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
     NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();

    // calculation of F_k(t) (= f^{-1}_k(t)) using initial SVD
    std::tie(Uk_, expKdiag_, Vk_) = this->hfm_.Lat.svd_hopping( hfm_.delta_);  // note Uk.expKdiag.Vk = expK
    PI_V_(NSL::Slice(0,Nt),NSL::Ellipsis()) = Vk_ * NSL::LinAlg::shift(this->hfm_.phiExp_,-tsrc).expand(Nx).transpose(1,2);
    PI_D_(NSL::Slice(0,Nt),NSL::Ellipsis()) = expKdiag_*NSL::LinAlg::exp( hfm_.mu_);
    PI_U_(NSL::Slice(0,Nt),NSL::Ellipsis()) = Uk_;

    SIGMA_V_(NSL::Slice(0,Nt),NSL::Ellipsis())=PI_V_(NSL::Slice(0,Nt),NSL::Ellipsis());
    SIGMA_D_(NSL::Slice(0,Nt),NSL::Ellipsis())=PI_D_(NSL::Slice(0,Nt),NSL::Ellipsis());
    SIGMA_U_(NSL::Slice(0,Nt),NSL::Ellipsis())=PI_U_(NSL::Slice(0,Nt),NSL::Ellipsis());


    if (!this->hfm_.stabilityMethod.compare("SVD")) {
      for (int t=1;t<Nt;t++) {
      	  // PI(t)=Fk(t)...Fk(0)
      	  vu_ = NSL::LinAlg::mat_mul(PI_V_(t,NSL::Ellipsis()),PI_U_(t-1,NSL::Ellipsis()));
      	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(PI_D_(t,NSL::Ellipsis())),vu_);
      	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(PI_D_(t-1,NSL::Ellipsis())));
      	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::svd(vu_);
      	  PI_U_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(t,NSL::Ellipsis()),uu_);
	  PI_D_(t,NSL::Ellipsis()) = dd_;
	  PI_V_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,PI_V_(t-1,NSL::Ellipsis()));

	  // SIGMA(t)=Fk(Nt-1)...Fk(t)
	  vu_ = NSL::LinAlg::mat_mul(SIGMA_V_(Nt-t,NSL::Ellipsis()),SIGMA_U_(Nt-1-t,NSL::Ellipsis()));
	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(SIGMA_D_(Nt-1-t,NSL::Ellipsis())));
	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(SIGMA_D_(Nt-t,NSL::Ellipsis())),vu_);
	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::svd(vu_);
	  SIGMA_U_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_U_(Nt-t,NSL::Ellipsis()),uu_);
	  SIGMA_D_(Nt-1-t,NSL::Ellipsis()) = dd_;
	  SIGMA_V_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,SIGMA_V_(Nt-1-t,NSL::Ellipsis()));
      }
   } else { //(!this->hfm_.stabilityMethod.compare("QR")) {
       for (int t=1;t<Nt;t++) {
      	  // PI(t)=Fk(t)...Fk(0)
      	  vu_ = NSL::LinAlg::mat_mul(PI_V_(t,NSL::Ellipsis()),PI_U_(t-1,NSL::Ellipsis()));
      	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(PI_D_(t,NSL::Ellipsis())),vu_);
      	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(PI_D_(t-1,NSL::Ellipsis())));
      	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::udt(vu_);
      	  PI_U_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(t,NSL::Ellipsis()),uu_);
	  PI_D_(t,NSL::Ellipsis()) = dd_;
	  PI_V_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,PI_V_(t-1,NSL::Ellipsis()));

	  // SIGMA(t)=Fk(Nt-1)...Fk(t)
	  vu_ = NSL::LinAlg::mat_mul(SIGMA_V_(Nt-t,NSL::Ellipsis()),SIGMA_U_(Nt-1-t,NSL::Ellipsis()));
	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(SIGMA_D_(Nt-1-t,NSL::Ellipsis())));
	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(SIGMA_D_(Nt-t,NSL::Ellipsis())),vu_);
	  std::tie( uu_,dd_,vv_ ) = NSL::LinAlg::udt(vu_);
	  SIGMA_U_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_U_(Nt-t,NSL::Ellipsis()),uu_);
	  SIGMA_D_(Nt-1-t,NSL::Ellipsis()) = dd_;
	  SIGMA_V_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,SIGMA_V_(Nt-1-t,NSL::Ellipsis()));
      }
  }

} // calcPiSigma(tsrc)


} // namespace NSL::Measure::Hubbard

#endif // NSL_FERMION_PROPAGATOR_SPIN_BASIS_TPP
