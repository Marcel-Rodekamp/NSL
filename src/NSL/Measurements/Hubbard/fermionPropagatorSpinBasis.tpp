#ifndef NSL_FERMION_PROPAGATOR_SPIN_BASIS_TPP
#define NSL_FERMION_PROPAGATOR_SPIN_BASIS_TPP

#include "Configuration/Configuration.tpp"
#include "concepts.hpp"
#include "device.tpp"
#include "parameter.tpp"
#include "../measure.hpp"

#include <algorithm>
#include <numeric>
#include <set>
#include <tuple>

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
                params["device"].to<NSL::Device>(),
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
            ),
            // wall sources are optional; without them measureK projects onto single
            // hopping eigenvectors, i.e. one band per block
            corrK_(
                params["device"].template to<NSL::Device>(),
                params["Nt"].template to<NSL::size_t>(),
                hasWallSources_(params) ? params["wallSources"].shape(1).template to<NSL::size_t>() : NSL::size_t(1), // bands
                hasWallSources_(params) ? params["wallSources"].shape(1).template to<NSL::size_t>() : NSL::size_t(1)  // bands
            ),
            phi_(
                params["device"].to<NSL::Device>(),
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
            ),
            basenode_(basenode_),
	invAp1F_U_(lattice.device(), 1, lattice.sites(), lattice.sites()),
	invAp1F_D_(lattice.device(), 1, lattice.sites()),
	invAp1F_T_(lattice.device(), 1, lattice.sites(), lattice.sites()),
	expKdiag_(lattice.device(), lattice.sites()),
	Uk_(lattice.device(), lattice.sites(), lattice.sites()),
	Vk_(lattice.device(), lattice.sites(), lattice.sites()),
	PI_U_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    PI_D_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites()),
    PI_V_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    SIGMA_U_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    SIGMA_D_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites()),
    SIGMA_V_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    Fk_U_(lattice.device(),lattice.sites(),lattice.sites()),
    Fk_D_(lattice.device(),lattice.sites()),   
	Fk_V_(lattice.device(),lattice.sites(),lattice.sites()),
	uu_(lattice.device(), lattice.sites(), lattice.sites()),
	dd_(lattice.device(), lattice.sites()),
	vv_(lattice.device(), lattice.sites(), lattice.sites()),
	vu_(lattice.device(), lattice.sites(), lattice.sites()),
	Qnew_(lattice.device(), lattice.sites(), lattice.sites()),
	Dnew_(lattice.device(), lattice.sites()),
	Vnew_(lattice.device(), lattice.sites(), lattice.sites()),
    Qnewt_(lattice.device(), params["Nt"].to<NSL::size_t>(), lattice.sites(), lattice.sites()),
	Dnewt_(lattice.device(), params["Nt"].to<NSL::size_t>(), lattice.sites()),
	Vnewt_(lattice.device(), params["Nt"].to<NSL::size_t>(), lattice.sites(), lattice.sites()),
    vut_(lattice.device(), params["Nt"].to<NSL::size_t>(), lattice.sites(), lattice.sites()),
    invAp1F_Ut_(lattice.device(), params["Nt"].to<NSL::size_t>(), lattice.sites(), lattice.sites()),
	invAp1F_Dt_(lattice.device(), params["Nt"].to<NSL::size_t>(), lattice.sites()),
	invAp1F_Tt_(lattice.device(), params["Nt"].to<NSL::size_t>(), lattice.sites(), lattice.sites()),
    Fk_Ut_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites(),lattice.sites()),
    Fk_Dt_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites()),   
	Fk_Vt_(lattice.device(),params["Nt"].to<NSL::size_t>(),lattice.sites(),lattice.sites()),
	uut_(lattice.device(), params["Nt"].to<NSL::size_t>(), lattice.sites(), lattice.sites()),
	ddt_(lattice.device(), params["Nt"].to<NSL::size_t>(), lattice.sites()),
	vvt_(lattice.device(), params["Nt"].to<NSL::size_t>(), lattice.sites(), lattice.sites()),
    Qnewtt_(lattice.device(), params["Nt"].to<NSL::size_t>()-1, lattice.sites(), lattice.sites()),
	Dnewtt_(lattice.device(), params["Nt"].to<NSL::size_t>()-1, lattice.sites()),
	Vnewtt_(lattice.device(), params["Nt"].to<NSL::size_t>()-1, lattice.sites(), lattice.sites())
    {}

    FermionPropagatorSpinBasis(LatticeType & lattice, NSL::Parameter params,NSL::H5IO & h5):
        FermionPropagatorSpinBasis(
            lattice,
            params,
            h5, 
            params["name"]
        )
    {}

    //! Calculate the \f( N_t \times N_x \times N_x \f) correlators, i.e.
    //! Propagators with averaged second time coordinate
    void measure() override;
    void measure(NSL::size_t NumberTimeSources, NSL::size_t config);
    void measureK();
    void measureK(NSL::size_t NumberTimeSources, NSL::size_t config);
    
    void measureDiagonal();
    void measureRow(NSL::size_t tsrc);
    void measureColumn(NSL::size_t tsrc);

    void calcPiSigma(NSL::size_t tsrc);

    // maybe make these private later. . .
    NSL::Tensor<Type> invAp1F_U_;
    NSL::Tensor<Type> invAp1F_D_;
    NSL::Tensor<Type> invAp1F_T_;
    NSL::Tensor<Type> expKdiag_, Uk_, Vk_;
    NSL::Tensor<Type> PI_U_; //(device,Nt,Nx,Nx); // stores U of M = U.D.V [ = Q.D.(D^{-1}.R) ]
    NSL::Tensor<Type> PI_D_; //(device,Nt,Nx);    // stores D of M = U.D.V
    NSL::Tensor<Type> PI_V_; //(device,Nt,Nx,Nx); // stores V of M = U.D.V
    NSL::Tensor<Type> SIGMA_U_; //(device,Nt,Nx,Nx); // stores U of M = U.D.V [ = Q.D.(D^{-1}.R) ]
    NSL::Tensor<Type> SIGMA_D_; //(device,Nt,Nx);    // stores D of M = U.D.V
    NSL::Tensor<Type> SIGMA_V_; //(device,Nt,Nx,Nx); // stores V of M = U.D.V
    NSL::Tensor<Type> Fk_U_; //(device,Nx,Nx);
    NSL::Tensor<Type> Fk_D_; //(device,Nx);   
    NSL::Tensor<Type> Fk_V_; //(device,Nx,Nx);
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
    NSL::Tensor<Type> invAp1F_Dt_;
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
    bool skip_(bool overwrite, std::string node){
        bool exists = this->h5_.exist(fmt::format("{}{}",std::string(basenode_),node));

        // if overwrite is specified always calculate the correlator
        if (overwrite){return false;}

        // if correlator doesn't exist always calculate it
        if (not exists){return false;}

        // if correlator exists only recompute if overwrite is true 
        // (this is the only remaining case)
        return true;
    }

    //! Project corr_ (Nt x Nx x Nx) onto all (sink,source) pairs of wall sources
    //! and write each (Nt x bands x bands) block to basenode_+node+"/kSink-kSrc".
    //! `sources` holds all wall-source vectors as rows, (kDim*bands) x Nx, row k*bands+sigma.
    void writeK_(const NSL::Tensor<Type> & sources, const std::string & node);

    //! true if params holds a non-empty wallSources tensor (blocks x bands x Nx)
    static bool hasWallSources_(NSL::Parameter & params){
        return params.contains("wallSources")
            && params["wallSources"].shape(0).template to<NSL::size_t>() > 0;
    }

    //! Without wall sources measureK projects onto eigenvectors of the hopping matrix K
    //! (see prepareEigenbasis_); computes them once, stores eigSources_ and the provenance
    void prepareEigenbasis_();

    NSL::Tensor<Type> corr_;
    NSL::Tensor<Type> corrK_;
    NSL::Tensor<Type> phi_;

    //! selected hopping eigenvectors as (nSel x 1 x Nx) blocks (undefined until prepareEigenbasis_)
    NSL::Tensor<Type> eigSources_;

    std::string basenode_;
};

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measure(NSL::size_t NumberTimeSources, NSL::size_t cfgID){

    // Reset memory
    // - Result correlator
    corr_ = Type(0);

    std::string node;

    bool PiSigma;

    // read configuration 
    this->h5_.read(phi_,fmt::format("{}/markovChain/{}/phi",std::string(basenode_),cfgID));

    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();

    NSL::size_t tsrcStep = ceil((Nt+0.0)/NumberTimeSources);

    for(NSL::size_t tsrc = 0; tsrc<Nt; tsrc+=tsrcStep){
        PiSigma = true;

        // first do up spin
	spin_ = NSL::Hubbard::Up;

	// calculate column
	node = fmt::format("/markovChain/{}/propagator/spinUp/invM[t,{}]",cfgID,tsrc); // node for column
	if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	   if (PiSigma) {
	      // populate the fermion matrix using the free configuration
    	      hfm_.populate(phi_,spin_);
	      calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	   }
	   measureColumn(tsrc);
	   this->h5_.write(corr_,std::string(basenode_)+node);  // write out the column
	}
	
	// calculate row
	node = fmt::format("/markovChain/{}/propagator/spinUp/invM[{},t]",cfgID,tsrc); // node for row
	if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	   if (PiSigma) {
	      // populate the fermion matrix using the free configuration
    	      hfm_.populate(phi_,spin_);
	      calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	   }
	   measureRow(tsrc);
	   this->h5_.write(corr_,std::string(basenode_)+node);  // write out the row
	}

	if(tsrc==0) { // calculate diagonal terms using tsrc=0 prefix/suffix terms
	   node = fmt::format("/markovChain/{}/propagator/spinUp/invM[t,t]",cfgID); // node for diagonal
	   if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	      if (PiSigma) {
	      	 // populate the fermion matrix using the free configuration
    	      	 hfm_.populate(phi_,spin_);
	      	 calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      	 PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	      }
	      measureDiagonal();
	      this->h5_.write(corr_,std::string(basenode_)+node);  // write out the diagonal
	  }
	}

	PiSigma = true; // reset bool for the downs. . .
	
	// now repeat for down spin
	spin_ = NSL::Hubbard::Down;
		
	// populate the fermion matrix using the free configuration
    	hfm_.populate(phi_,spin_);

	// calculate column
	node = fmt::format("/markovChain/{}/propagator/spinDown/invM[t,{}]",cfgID,tsrc); // node for column
	if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	   if (PiSigma) {
	      // populate the fermion matrix using the free configuration
    	      hfm_.populate(phi_,spin_);
	      calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	   }
	   measureColumn(tsrc);
	   this->h5_.write(corr_,std::string(basenode_)+node);  // write out the column
	}

	// calculate row
	node = fmt::format("/markovChain/{}/propagator/spinDown/invM[{},t]",cfgID,tsrc); // node for row
	if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	   if (PiSigma) {
	      // populate the fermion matrix using the free configuration
    	      hfm_.populate(phi_,spin_);
	      calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	   }
	   measureRow(tsrc);
	   this->h5_.write(corr_,std::string(basenode_)+node);  // write out the row
	}

	if(tsrc==0) { // calculate diagonal terms using tsrc=0 prefix/suffix terms
	   node = fmt::format("/markovChain/{}/propagator/spinDown/invM[t,t]",cfgID); // node for diagonal
	   if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
	      if (PiSigma) {
	      	 // populate the fermion matrix using the free configuration
    	      	 hfm_.populate(phi_,spin_);
	      	 calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
	      	 PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
	      }
	      measureDiagonal();
	      this->h5_.write(corr_,std::string(basenode_)+node);  // write out the diagonal
	   }
      }

    } // tsrc
      
} // measure(Ntsrc,config);

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measure(){
    NSL::Logger::info("Start Measuring Hubbard::FermionPropagatorSpinBasis");

    // This is the default basenode we used so far
    // ToDo: this should go into the const

    // write the non interacting correlator
    std::string nodeD,nodeC,nodeR;
    spin_ = NSL::Hubbard::Up;
    nodeD = "/NonInteracting/propagator/spinUp/invM[t,t]";
    nodeC = "/NonInteracting/propagator/spinUp/invM[t,0]";
    nodeR = "/NonInteracting/propagator/spinUp/invM[0,t]";
    // this is a shortcut, we don't need to calculate the non-interacting 
    // correlators if we won't update the file
    if(!skip_(this->params_["overwrite"],nodeD)) {
        // measure the non-interacting theory
        // U = 0 <=> phi = 0
        phi_ = Type(0);
	hfm_.populate(phi_,spin_);
	calcPiSigma(1); // calculuate prefix/suffix terms with tsrc
	measureDiagonal();
	h5_.write(corr_,std::string(basenode_)+nodeD);  // write out the diagonal
	measureRow(1);
	h5_.write(corr_,std::string(basenode_)+nodeR);  // write out the row
	measureColumn(1);
	h5_.write(corr_,std::string(basenode_)+nodeC);  // write out the column
     
    } else {
        NSL::Logger::info("Non-interacting invM[t,t] already exists");
    }

    spin_ = NSL::Hubbard::Down;
    nodeD = "/NonInteracting/propagator/spinDown/invM[t,t]";
    nodeC = "/NonInteracting/propagator/spinDown/invM[t,0]";
    nodeR = "/NonInteracting/propagator/spinDown/invM[0,t]";
    // this is a shortcut, we don't need to calculate the non-interacting
    // correlators if we won't update the file
    if(!skip_(this->params_["overwrite"],nodeD)) {
        // measure the non-interacting theory
        // U = 0 <=> phi = 0
        phi_ = Type(0);
        hfm_.populate(phi_,spin_);
        calcPiSigma(1); // calculuate prefix/suffix terms with tsrc
        measureDiagonal();
        h5_.write(corr_,std::string(basenode_)+nodeD);  // write out the diagonal
        measureRow(1);
        h5_.write(corr_,std::string(basenode_)+nodeR);  // write out the row
        measureColumn(1);
        h5_.write(corr_,std::string(basenode_)+nodeC);  // write out the column

    } else {
        NSL::Logger::info("Non-interacting invM[t,t] already exists");
    }


    // Interacting Correlators
    // Initialize memory for the configurations
     // get the range of configuration ids from the h5file
    auto [minCfg, maxCfg] = this->h5_.getMinMaxConfigs(std::string(basenode_)+"/markovChain");
    NSL::size_t saveFreq = this->params_["save frequency"];
    
    NSL::Logger::info("Found trajectories: {} to {} with save frequency {}",
        minCfg, maxCfg, saveFreq
    );

    for (NSL::size_t cfgID = minCfg; cfgID<=maxCfg; ++cfgID){

        NSL::Logger::info("Calculating Correlator on {}/{}", cfgID, maxCfg);

        // compute the correlator for the differen time sources
        measure(this->params_["Number Time Sources"], cfgID);

    } // for cfgID

} // measure()

//! Hopping-eigenbasis projection, used by measureK when no wall sources are given.
//!
//! Selection: eigenpairs (lambda_n, v_n) of the hopping matrix K (eigh, ascending lambda,
//! index n as in numpy.linalg.eigh; mu is not included, so the orbitals are the same for
//! every mu). Order all n by (|lambda|, lambda, n), where values that agree within
//! 1e-8*max(1,max|lambda|) count as equal, and take the first two (smallest |lambda|)
//! and the last two (largest |lambda|). For Nx < 4 these sets overlap; each eigenvector is
//! used once, i.e. all Nx eigenvectors are selected. The selected vectors are ordered by
//! ascending (lambda, n); vector j becomes block j with one band, so measureK writes it at
//! k = 2j and its complex conjugate at k = 2j+1 (exactly as for wall sources).
//! If the selection cuts through a degenerate eigenspace a warning is logged: the result
//! then depends on the eigensolver's basis of that subspace.
//!
//! Provenance (written once, then reused): <basenode>/Eigenbasis/
//!   spectrum     (Nx)       all eigenvalues of K, ascending
//!   indices      (nSel)     eigh index n of block j
//!   eigenvalues  (nSel)     lambda of block j
//!   eigenvectors (Nx,nSel)  column j = vector of block j (sign/phase as used)
//! If the file already holds an eigenbasis it is checked against K and reused, so all
//! configurations of a file share the same eigenvector signs.
template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::prepareEigenbasis_(){
    if (eigSources_.defined()) { return; }

    const NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();
    const NSL::Device device = this->params_["device"].template to<NSL::Device>();
    const std::string base = std::string(basenode_) + "/Eigenbasis";

    // eigen decomposition of K on the CPU; real symmetric K gives real eigenvectors
    torch::Tensor K = static_cast<torch::Tensor>(this->hfm_.Lat.hopping_matrix()).to(torch::kCPU).to(torch::kComplexDouble);
    torch::Tensor evals, evecs;
    if (torch::imag(K).abs().max().template item<double>() == 0.) {
        std::tie(evals, evecs) = torch::linalg_eigh(torch::real(K).contiguous(), "L");
        evecs = evecs.to(torch::kComplexDouble);
    } else {
        std::tie(evals, evecs) = torch::linalg_eigh(K, "L");
    }
    evals = evals.to(torch::kDouble).contiguous();
    std::vector<double> lambda(evals.data_ptr<double>(), evals.data_ptr<double>()+Nx);

    double maxAbs = 0.;
    for (double l: lambda) { maxAbs = std::max(maxAbs, std::abs(l)); }
    const double tolDeg = 1e-8 * std::max(1., maxAbs);

    // order by (|lambda|, lambda, n) and select the 2 smallest and the 2 largest |lambda|.
    // Values equal up to tolDeg (e.g. the +-lambda pairs of a bipartite lattice, which differ
    // only by rounding) must tie, so |lambda| and lambda are first mapped to cluster ids of
    // consecutive (within tolDeg) sorted values; the comparison then uses these exact integers.
    auto clusters = [&](auto value){
        std::vector<NSL::size_t> byValue(Nx), id(Nx);
        std::iota(byValue.begin(), byValue.end(), 0);
        std::stable_sort(byValue.begin(), byValue.end(), [&](NSL::size_t a, NSL::size_t b){ return value(a) < value(b); });
        NSL::size_t c = 0;
        for (NSL::size_t i = 0; i < Nx; i++) {
            if (i > 0 && value(byValue[i]) - value(byValue[i-1]) > tolDeg) { c++; }
            id[byValue[i]] = c;
        }
        return id;
    };
    const std::vector<NSL::size_t> absId = clusters([&](NSL::size_t n){ return std::abs(lambda[n]); });
    const std::vector<NSL::size_t> lamId = clusters([&](NSL::size_t n){ return lambda[n]; });
    std::vector<NSL::size_t> order(Nx);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](NSL::size_t a, NSL::size_t b){
        return std::make_tuple(absId[a], lamId[a], a) < std::make_tuple(absId[b], lamId[b], b);
    });
    std::set<NSL::size_t> selSet;
    for (NSL::size_t i = 0; i < std::min<NSL::size_t>(2,Nx); i++) { selSet.insert(order[i]); selSet.insert(order[Nx-1-i]); }
    std::vector<NSL::size_t> sel(selSet.begin(), selSet.end()); // ascending n == ascending (lambda, n)
    const NSL::size_t nSel = sel.size();

    std::string desc;
    for (NSL::size_t j = 0; j < nSel; j++) {
        desc += fmt::format("{}block {} (k={},{}): n={} lambda={:.6f}", j ? "; " : "", j, 2*j, 2*j+1, sel[j], lambda[sel[j]]);
    }
    NSL::Logger::info("Hopping eigenbasis: {} of {} eigenvectors selected (2 smallest + 2 largest |lambda|): {}", nSel, Nx, desc);
    if (Nx < 4) {
        NSL::Logger::warn("Hopping eigenbasis: Nx = {} < 4, all {} eigenvectors are used", Nx, nSel);
    }

    // degeneracy checks (one message per selected level):
    //  - an unselected eigenvector with the same lambda: the cut splits that eigenspace, so the
    //    selection depends on the eigensolver's basis of it -> warning
    //  - an unselected eigenvector with the same |lambda| but opposite lambda: tie resolved by lambda -> info
    std::set<NSL::size_t> reported;
    for (NSL::size_t m: sel) {
        std::string sameLevel, sameAbs;
        for (NSL::size_t n = 0; n < Nx; n++) {
            if (selSet.contains(n)) { continue; }
            if (std::abs(lambda[n] - lambda[m]) <= tolDeg) { sameLevel += fmt::format(" {}", n); }
            else if (std::abs(std::abs(lambda[n]) - std::abs(lambda[m])) <= tolDeg) { sameAbs += fmt::format(" {}", n); }
        }
        // report each level once (selected members of the same level share the message)
        bool first = true;
        for (NSL::size_t r: reported) { if (std::abs(lambda[r] - lambda[m]) <= tolDeg) { first = false; } }
        if (!first) { continue; }
        reported.insert(m);
        std::string selLevel;
        for (NSL::size_t n: sel) { if (std::abs(lambda[n] - lambda[m]) <= tolDeg) { selLevel += fmt::format(" {}", n); } }
        if (!sameLevel.empty()) {
            NSL::Logger::warn("Hopping eigenbasis: the selection splits the degenerate eigenspace lambda={:.6f} (selected n:{}; not selected n:{}); the projected propagators depend on the eigensolver's basis of that subspace", lambda[m], selLevel, sameLevel);
        } else if (selLevel.size() > fmt::format(" {}", m).size()) {
            NSL::Logger::info("Hopping eigenbasis: degenerate level lambda={:.6f} fully selected (n:{}); the individual blocks are basis dependent", lambda[m], selLevel);
        }
        if (!sameAbs.empty()) {
            NSL::Logger::info("Hopping eigenbasis: |lambda|={:.6f} tie between selected n:{} and not selected n:{} resolved by lambda", std::abs(lambda[m]), selLevel, sameAbs);
        }
    }

    // selected eigenpairs: V (Nx x nSel), columns = vectors of the blocks
    torch::Tensor idx = torch::tensor(std::vector<int64_t>(sel.begin(), sel.end()), torch::kLong);
    torch::Tensor V = evecs.index_select(1, idx).contiguous();
    torch::Tensor lamSel = evals.index_select(0, idx).contiguous();

    if (this->h5_.exist(base+"/eigenvectors")) {
        // reuse the stored eigenbasis (keeps the eigenvector signs of this file) after checking it
        NSL::Tensor<Type> Vstored; NSL::Tensor<double> lamStored;
        this->h5_.read(Vstored, base+"/eigenvectors");
        this->h5_.read(lamStored, base+"/eigenvalues");
        torch::Tensor Vs = static_cast<torch::Tensor>(Vstored).to(torch::kCPU).to(torch::kComplexDouble);
        torch::Tensor ls = static_cast<torch::Tensor>(lamStored).to(torch::kCPU).to(torch::kDouble);
        bool ok = Vs.dim() == 2 && Vs.size(0) == (int64_t) Nx && Vs.size(1) == (int64_t) nSel && ls.numel() == (int64_t) nSel;
        if (ok) {
            ok = (ls - lamSel).abs().max().template item<double>() <= tolDeg
              && (torch::matmul(K, Vs) - Vs * ls.to(torch::kComplexDouble)).abs().max().template item<double>() <= tolDeg
              && (torch::matmul(Vs.conj().transpose(0,1), Vs) - torch::eye(nSel, torch::kComplexDouble)).abs().max().template item<double>() <= 1e-8;
        }
        if (!ok) {
            NSL::Logger::error("Hopping eigenbasis stored in {} does not match the lattice of this run", base);
            throw std::runtime_error("FermionPropagatorSpinBasis: stored hopping eigenbasis does not match the lattice");
        }
        NSL::Logger::info("Hopping eigenbasis: using the eigenvectors stored in {}", base);
        V = Vs;
    } else {
        NSL::Tensor<double> spectrum(evals.clone());
        NSL::Tensor<double> lamOut(lamSel.clone());
        NSL::Tensor<int> idxOut(idx.to(torch::kInt).clone());
        NSL::Tensor<Type> Vout(V.clone());
        this->h5_.write(spectrum, base+"/spectrum");
        this->h5_.write(idxOut,   base+"/indices");
        this->h5_.write(lamOut,   base+"/eigenvalues");
        this->h5_.write(Vout,     base+"/eigenvectors");
    }

    // blocks (nSel x 1 x Nx): block j holds column j of V
    eigSources_ = NSL::Tensor<Type>(V.transpose(0,1).reshape({(int64_t) nSel, 1, (int64_t) Nx}).contiguous().to(device.device().device()));
} // prepareEigenbasis_()

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::writeK_(const NSL::Tensor<Type> & sources, const std::string & node){
    int bDim = corrK_.shape(1);
    int kDim = sources.shape(0)/bDim;

    // all projections at once: proj(t,i,j) = sum_{x,y} conj(sources(i,x)) corr_(t,x,y) sources(j,y)
    // with i = kSink*bDim+sigmaSink, j = kSrc*bDim+sigmaSrc
    NSL::Tensor<Type> proj = NSL::LinAlg::mat_mul(
        NSL::LinAlg::conj(sources),
        NSL::LinAlg::mat_mul(corr_, NSL::LinAlg::transpose(sources))
    ); // (Nt, kDim*bDim, kDim*bDim)

    for (int kSrc=0; kSrc<kDim; kSrc++ ) {
        for (int kSink=0; kSink<kDim; kSink++) {
            // deep copy into the contiguous corrK_ (Nt x bands x bands) buffer
            corrK_ = proj(NSL::Slice(), NSL::Slice(kSink*bDim,(kSink+1)*bDim), NSL::Slice(kSrc*bDim,(kSrc+1)*bDim));
            std::string leaf = fmt::format("/{}-{}",kSink,kSrc);
            this->h5_.write(corrK_,std::string(basenode_)+node+leaf);
        }
    } // for kSrc
} // writeK_(sources,node)

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureK(NSL::size_t NumberTimeSources, NSL::size_t cfgID){
    const bool wallMode = hasWallSources_(this->params_);
    if (!wallMode) { prepareEigenbasis_(); }

    // Reset memory
    // - Result correlator
    corr_ = Type(0);
    corrK_ = Type(0);

    std::string node;
    std::string prefix;

    bool PiSigma;

    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();

    NSL::size_t tsrcStep = ceil((Nt+0.0)/NumberTimeSources);

    // source blocks (blocks x bands x Nx): the wall sources, or else the selected
    // hopping eigenvectors (one band each)
    NSL::Tensor<Type> blocks = wallMode ? NSL::Tensor<Type> (params_["wallSources"]) : eigSources_;

    // wall sources: even k = source, odd k = conj(source)
    NSL::Tensor<Type> wallSources(params_["device"].to<NSL::Device>(), 2 * blocks.shape(0), blocks.shape(1), blocks.shape(2));
    wallSources(NSL::Slice(0,NSL::none_t(), 2), NSL::Ellipsis()) = blocks;
    wallSources(NSL::Slice(1,NSL::none_t(), 2), NSL::Ellipsis()) = NSL::LinAlg::conj(blocks);
    // flatten to (kDim*bands) x Nx: row k*bands+sigma is wall source (k,sigma)
    wallSources.reshape(wallSources.shape(0)*wallSources.shape(1), wallSources.shape(2));

    // if non-interacting (phi = 0) write to /NonInteracting, otherwise to /markovChain/cfgID
    if ((phi_ == Type(0)).all()) {
        prefix = "/NonInteracting/propagator";
    } else {
        prefix = fmt::format("/markovChain/{}/propagator",cfgID);
    }

    const NSL::Hubbard::Spin spins[2] = {NSL::Hubbard::Up, NSL::Hubbard::Down};
    const std::string spinNames[2] = {"spinUp", "spinDown"};

    for(NSL::size_t tsrc = 0; tsrc<Nt; tsrc+=tsrcStep){
        for(int s = 0; s<2; s++){
            PiSigma = true;
            spin_ = spins[s];

            // calculate column
            node = fmt::format("{}/{}/invM[t,{}]",prefix,spinNames[s],tsrc); // node for column
            if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
                if (PiSigma) {
                    // populate the fermion matrix using the configuration
                    hfm_.populate(phi_,spin_);
                    calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
                    PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
                }
                measureColumn(tsrc);
                writeK_(wallSources,node);  // write out the projected column
            }else{
                NSL::Logger::info("Config #{} already has {} columns, skipping... ", cfgID, spinNames[s]);
            }

            // calculate row
            node = fmt::format("{}/{}/invM[{},t]",prefix,spinNames[s],tsrc); // node for row
            if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
                if (PiSigma) {
                    // populate the fermion matrix using the configuration
                    hfm_.populate(phi_,spin_);
                    calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
                    PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
                }
                measureRow(tsrc);
                writeK_(wallSources,node);  // write out the projected row
            }else{
                NSL::Logger::info("Config #{} already has {} rows, skipping... ", cfgID, spinNames[s]);
            }

            if(tsrc==0) { // calculate diagonal terms using tsrc=0 prefix/suffix terms
                node = fmt::format("{}/{}/invM[t,t]",prefix,spinNames[s]); // node for diagonal
                if (!skip_(this->params_["overwrite"],node)) { // if node already exists (meaning this is already calculated), then skip
                    if (PiSigma) {
                        // populate the fermion matrix using the configuration
                        hfm_.populate(phi_,spin_);
                        calcPiSigma(tsrc); // calculuate prefix/suffix terms with tsrc
                        PiSigma = false;  // set flag to false to avoid repeating this calculation for the other terms
                    }
                    measureDiagonal();
                    writeK_(wallSources,node);  // write out the projected diagonal
                }else{
                    NSL::Logger::info("Config #{} already has {} diagonal, skipping... ", cfgID, spinNames[s]);
                }
            }
        } // spin
    } // tsrc

} // measureK(Ntsrc,config)

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureK(){
    NSL::Logger::info("Start Measuring Momentum Hubbard::FermionPropagatorSpinBasis");
    if (hasWallSources_(this->params_)) {
        NSL::Logger::info("Projecting onto {} wall-source block(s)", this->params_["wallSources"].shape(0).template to<NSL::size_t>());
    } else {
        NSL::Logger::info("No wall sources given: projecting onto eigenvectors of the hopping matrix");
        prepareEigenbasis_();
    }

    // write the momenta out
    if (!h5_.exist(std::string(basenode_)+"/Momenta")){
        NSL::Tensor<double> momenta = params_["momenta"];
        this->h5_.write(momenta,std::string(basenode_)+"/Momenta");
    }

    // write the non interacting correlator 
    std::string node = "/NonInteracting/propagator";
    // this is a shortcut, we don't need to calculate the non-interacting 
    // correlators if we won't update the file
    if(!skip_(this->params_["overwrite"],node)) {
        // measure the non-interacting theory
        // U = 0 <=> phi = 0
        phi_ = Type(0);

        measureK(1,0);
    }else{
        NSL::Logger::info("Non-interacting propagator already exists");
    }

    // Interacting Correlators
    // get the range of configuration ids from the h5file
    auto [minCfg, maxCfg] = this->h5_.getMinMaxConfigs(std::string(basenode_)+"/markovChain");
    NSL::size_t saveFreq = this->params_["save frequency"];
    
    NSL::Logger::info("Found trajectories: {} to {} with save frequency {}",
        minCfg, maxCfg, saveFreq
    );

    for (NSL::size_t cfgID = minCfg; cfgID<=maxCfg; ++cfgID){

        // configuration ids need not be contiguous (save frequency, and getMinMaxConfigs
        // currently reports minCfg = 0); a failed read would leave the previous phi_ in
        // place and write its propagator under this cfgID, so skip missing configurations
        std::string phiNode = fmt::format("{}/markovChain/{}/phi",std::string(basenode_),cfgID);
        if (!this->h5_.exist(phiNode)){ continue; }

        NSL::Logger::info("Calculating Correlator on {}/{}", cfgID, maxCfg);

        // read configuration 
        this->h5_.read(phi_,phiNode);

        // compute the correlator for the different time sources
        measureK(this->params_["Number Time Sources"], cfgID);
    } // for cfgID

} // measureK()

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureDiagonal(){
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
    NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();
    const NSL::Device device = this->phi_.device();

    // t=0 term
    std::tie( Qnew_, Dnew_, Vnew_ ) = NSL::LinAlg::udt( NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(0,NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(SIGMA_D_(0,NSL::Ellipsis())));

    invAp1F_Ut_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnew_));
    invAp1F_Tt_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));

    corr_(0,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(invAp1F_Ut_(0,NSL::Ellipsis()),invAp1F_Tt_(0,NSL::Ellipsis()));



    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_V_(NSL::Slice(0,Nt-1),NSL::Ellipsis()),SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()));
	vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::diag_embed(PI_D_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),vut_(NSL::Slice(1,Nt),NSL::Ellipsis()));
	vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(SIGMA_D_(NSL::Slice(1,Nt),NSL::Ellipsis())));


    // std::cout << "gets here 1" << std::endl;
    if (!this->hfm_.stabilityMethod.compare("SVD")) {
	   std::tie( uut_,ddt_,vvt_ ) = NSL::LinAlg::svd(vut_);
	} else { //(!this->hfm_.stabilityMethod.compare("QR")) {
	   std::tie( uut_,ddt_,vvt_ ) = NSL::LinAlg::udt(vut_);
	}


    // std::cout << "gets here 2" << std::endl;
    Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()),uut_(NSL::Slice(1,Nt),NSL::Ellipsis()));
    Fk_Dt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = ddt_(NSL::Slice(1,Nt),NSL::Ellipsis());
	Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vvt_(NSL::Slice(1,Nt),NSL::Ellipsis()),SIGMA_V_(NSL::Slice(1,Nt),NSL::Ellipsis()));

    // std::cout << "gets here 3" << std::endl;
    std::tie( Qnewtt_, Dnewtt_, Vnewtt_ ) = NSL::LinAlg::udt( NSL::LinAlg::solve(Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis()), NSL::LinAlg::adjoint(Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(Fk_Dt_(NSL::Slice(1,Nt),NSL::Ellipsis())));

    // std::cout << "gets here 4" << std::endl;
    invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnewtt_, Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnewtt_));
    invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()), Qnewtt_ ));
    // std::cout << "gets here 5" << std::endl;

	corr_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(invAp1F_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()),invAp1F_Tt_(NSL::Slice(1,Nt),NSL::Ellipsis()));


} // measureDiagonal()

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::measureRow(NSL::size_t tsrc){
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
    NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();
    const NSL::Device device = this->phi_.device();


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

/***
    // Nt/2<t<Nt
    vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::solve(SIGMA_V_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()), NSL::LinAlg::adjoint(PI_U_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis())), false ),NSL::LinAlg::diag_embed(1./SIGMA_D_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())));
	vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()),NSL::LinAlg::solve(PI_V_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())), false ));
	vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())+ NSL::LinAlg::diag_embed(PI_D_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()));

    // 1<t<Nt/2 
    vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_V_(NSL::Slice(1,Nt/2),NSL::Ellipsis()),PI_U_(NSL::Slice(0,Nt/2-1),NSL::Ellipsis()));
	vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()),NSL::LinAlg::diag_embed(PI_D_(NSL::Slice(0,Nt/2-1),NSL::Ellipsis())));
	vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()),PI_V_(NSL::Slice(0,Nt/2-1),NSL::Ellipsis()));
	vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()),SIGMA_U_(NSL::Slice(1,Nt/2),NSL::Ellipsis()));
	vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis())+ NSL::LinAlg::diag_embed(1./SIGMA_D_(NSL::Slice(1,Nt/2),NSL::Ellipsis()));



    if (!this->hfm_.stabilityMethod.compare("QR")) {
	   std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::udt(vut_);
	} else if (!this->hfm_.stabilityMethod.compare("SVD")) {
	   std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::svd(vut_);
	}



    // Nt/2<t<Nt
    invAp1F_Ut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()), PI_V_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())));
    invAp1F_Tt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( PI_U_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()), Qnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) ));

    // 1<t<Nt/2
    if (!this->hfm_.stabilityMethod.compare("QR")) {
	   invAp1F_Ut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::solve_triangular(Vnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis()), SIGMA_U_(NSL::Slice(1,Nt/2),NSL::Ellipsis()), false),NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis())));
	} else if (!this->hfm_.stabilityMethod.compare("SVD")) {
	   invAp1F_Ut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_U_(NSL::Slice(1,Nt/2),NSL::Ellipsis()),NSL::LinAlg::adjoint(Vnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis()))); 
	}
    invAp1F_Tt_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::adjoint(Qnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis())), SIGMA_V_(NSL::Slice(1,Nt/2),NSL::Ellipsis()));
***/

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
    NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();
    const NSL::Device device = this->phi_.device();


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

/***
// 1<t<Nt/2
    vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::solve(PI_V_(NSL::Slice(0,Nt/2-1),NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(NSL::Slice(1,Nt/2),NSL::Ellipsis())), false ),NSL::LinAlg::diag_embed(1./PI_D_(NSL::Slice(0,Nt/2-1),NSL::Ellipsis())));
	vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()),NSL::LinAlg::solve(SIGMA_V_(NSL::Slice(1,Nt/2),NSL::Ellipsis()), NSL::LinAlg::adjoint(PI_U_(NSL::Slice(0,Nt/2-1),NSL::Ellipsis())), false ));
	vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = vut_(NSL::Slice(1,Nt/2),NSL::Ellipsis())+ NSL::LinAlg::diag_embed(SIGMA_D_(NSL::Slice(1,Nt/2),NSL::Ellipsis()));

    // Nt/2<t<Nt
    vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_V_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()),SIGMA_U_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()));
	vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(SIGMA_D_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())));
	vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()),SIGMA_V_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()));
	vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()),PI_U_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()));
	vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = vut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())+ NSL::LinAlg::diag_embed(1./PI_D_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()));



    if (!this->hfm_.stabilityMethod.compare("QR")) {
	   std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::udt(vut_);
	} else if (!this->hfm_.stabilityMethod.compare("SVD")) {
	   std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::svd(vut_);
	}
    

    
    // 1<t<Nt/2
    invAp1F_Ut_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis()), SIGMA_V_(NSL::Slice(1,Nt/2),NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis())));
    invAp1F_Tt_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(NSL::Slice(1,Nt/2),NSL::Ellipsis()), Qnewt_(NSL::Slice(1,Nt/2),NSL::Ellipsis()) ));

    // Nt/2<t<Nt
    if (!this->hfm_.stabilityMethod.compare("QR")) {
	   invAp1F_Ut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::solve_triangular(Vnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()), PI_U_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()), false),NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())));
	} else if (!this->hfm_.stabilityMethod.compare("SVD")) {
	   invAp1F_Ut_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(PI_U_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()),NSL::LinAlg::adjoint(Vnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()))),NSL::LinAlg::diag_embed(1./Dnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())));
	}
    invAp1F_Tt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::adjoint(Qnewt_(NSL::Slice(Nt/2,Nt),NSL::Ellipsis())), PI_V_(NSL::Slice(Nt/2-1,Nt-1),NSL::Ellipsis()));
***/

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
