#ifndef NSL_FERMION_PROPAGATOR_SPIN_BASIS_TPP
#define NSL_FERMION_PROPAGATOR_SPIN_BASIS_TPP

#include "Configuration/Configuration.tpp"
#include "concepts.hpp"
#include "device.tpp"
#include "parameter.tpp"
#include "../measure.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <tuple>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

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
            // wall sources are optional; without them measureK projects onto one block of
            // hopping eigenvectors (setupProjection_ resizes corrK_ to their number of bands)
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
    {
        configureAllocator_();
    }

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

    //! UDT decomposition t = U.D.V (NSL::LinAlg::udt: QR, D = diag(R), V = D^{-1} R) computed with the
    //! CPU (LAPACK) QR also when t lives on the GPU. The matrices factorised here span |x| ~ 1e-170 ..
    //! 1e170 (beta = 90, mu != 0). torch's batched CUDA QR (cuSOLVER/cuBLAS path) returns NON-FINITE
    //! R for finite inputs with |x| >~ 1e154 = sqrt(DBL_MAX), and also for row/column-graded inputs
    //! spanning ~1e-170..1e170, while LAPACK, the single-matrix CUDA QR and MAGMA stay finite. The
    //! fix is about finiteness, not accuracy: on such graded matrices |diag(R)| from LAPACK (and from
    //! the single-matrix CUDA QR) also differs from an exact QR by huge factors. With the host QR,
    //! max|GPU - CPU| / max|node| <~ 1e-10 (measured on 32 perylene configs: 3.8e-11 at mu = 0,
    //! 6.5e-11 at mu = 0.4062, 1.4e-10 at mu = 0.4688). Against a double-precision sparse-LU
    //! reference, normalised per time slice, CPU and GPU both agree to <= ~2e-10; element-wise
    //! relative errors reach ~3e-7..6e-7 for elements >= 1e-6 of the time-slice maximum (inherent
    //! precision of the stabilised algorithm at beta = 90).
    //! Cost: the host round trip is free for perylene (Nx = 20) but slows the GPU run down for
    //! Nx >~ 100 (measured vs 7b495d0: tube Nx=112/Nt=256 1.6x, ring Nx=200/Nt=128 2.1x).
    //! Not changed here: the batched CUDA linalg_solve can also go non-finite for row-graded
    //! operands (the operands NSL passes stayed finite in all tests), and other code still calls
    //! NSL::LinAlg::udt on batched (per-time-slice) device tensors and can hit the same batched CUDA
    //! QR overflow on the GPU: the charge-basis propagator (fermionPropagator.tpp: measureDiagonal,
    //! measureRow, measureColumn) and the HMC force gradLogDetM in hubbardExp.tpp and
    //! hubbardExpSpinBasis.tpp. On the CPU this is exactly NSL::LinAlg::udt.
    static std::tuple<NSL::Tensor<Type>,NSL::Tensor<Type>,NSL::Tensor<Type>> udtHost_(const NSL::Tensor<Type> & t){
        torch::Tensor M = static_cast<torch::Tensor>(t);
        if (!M.is_cuda()) { return NSL::LinAlg::udt(t); }
        auto [Q, D, V] = NSL::LinAlg::udt(NSL::Tensor<Type>(M.cpu()));
        return std::make_tuple( NSL::Tensor<Type>(static_cast<torch::Tensor>(Q).to(M.device())),
                                NSL::Tensor<Type>(static_cast<torch::Tensor>(D).to(M.device())),
                                NSL::Tensor<Type>(static_cast<torch::Tensor>(V).to(M.device())) );
    }

    //! Stop (exception, nothing written for this node) if a result is not finite
    void requireFinite_(const NSL::Tensor<Type> & x, const std::string & node){
        if (!torch::isfinite(static_cast<torch::Tensor>(x)).all().template item<bool>()) {
            NSL::Logger::error("FermionPropagatorSpinBasis: non-finite values in {}; nothing written for this node", node);
            throw std::runtime_error("FermionPropagatorSpinBasis: non-finite propagator values in " + node);
        }
    }

    //! write corr_ to node after checking that it is finite
    void writeFinite_(NSL::Tensor<Type> & x, const std::string & node){
        requireFinite_(x, node);
        this->h5_.write(x, node);
    }

    //! Project corr_ (Nt x Nx x Nx) onto all (sink,source) pairs of source blocks
    //! and write each (Nt x bands x bands) block to basenode_+node+"/kSink-kSrc".
    //! `sources` holds all source vectors as rows, (kDim*bands) x Nx, row k*bands+sigma
    //! (wall mode: kDim = 2*blocks with conjugate copies; eigenbasis: kDim = 1, bands = m).
    void writeK_(const NSL::Tensor<Type> & sources, const std::string & node);

    //! true if params holds a non-empty wallSources tensor (blocks x bands x Nx)
    static bool hasWallSources_(NSL::Parameter & params){
        return params.contains("wallSources")
            && params["wallSources"].shape(0).template to<NSL::size_t>() > 0;
    }

    //! Choose the projection of measureK (wall sources, or else eigenvectors of the hopping
    //! matrix K: params["eigenvalueIndices"] or the default selection), check it against what the
    //! file already holds and store mode marker and eigenbasis provenance (see the definition);
    //! runs once, before anything is written
    void setupProjection_();
    bool projectionReady_ = false;

    NSL::Tensor<Type> corr_;
    NSL::Tensor<Type> corrK_;
    NSL::Tensor<Type> phi_;

    //! selected hopping eigenvectors as the m bands of one block, (m x Nx), row j = band j
    //! (undefined until setupProjection_)
    NSL::Tensor<Type> eigSources_;

    //! Tr_x G(0,0) of the full equal-time propagator (the t = 0 term of measureDiagonal) for the
    //! prefix/suffix products computed by calcPiSigma(0); same definition as the Q of earlier
    //! measurements (luu: markovChain/<cfg>/measurements/<spin>/Q)
    Type traceG00_();

    std::string basenode_;

    //! glibc: use a fixed mmap threshold of 128 KiB for the whole process. Every configuration writes
    //! many small HDF5 datasets whose long-lived allocations stay in the heap; with glibc's default
    //! dynamic threshold (raised after large frees) the large tensor temporaries (Nt x Nx x Nx) are
    //! then also placed in the heap between them, so the freed tensor memory can neither be reused nor
    //! returned and the resident set grew by several MB per configuration (perylene, Nt = 512) until
    //! the node ran out of memory. With a fixed threshold, allocations >= 128 KiB are mmapped and
    //! returned to the OS on free. (mallopt also disables the dynamic adjustment.) No-op elsewhere.
    //! The setting is process-wide. 128 KiB was the only tested threshold that removes the
    //! fragmentation (256 KiB and 1 MiB fragment partially, 4 MiB behaves like no fix); it costs
    //! ~+10% run time in eigenbasis mode and ~+14% in wall mode.
    //! Not addressed: while the h5 file is open, HDF5 (1.14.5) keeps ~47-67 B of live heap per
    //! created dataset (open-file state, released when the file is closed), growing linearly: ~0.11 MB
    //! per configuration in eigenbasis mode with the current per-vector layout (~2.6 GB per process
    //! after 19k configurations) and ~5 kB per configuration in wall mode.
    static void configureAllocator_(){
#if defined(__GLIBC__)
        static const bool configured = (mallopt(M_MMAP_THRESHOLD, 128 * 1024) == 1);
        (void) configured;
#endif
    }
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
	   writeFinite_(corr_,std::string(basenode_)+node);  // write out the column
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
	   writeFinite_(corr_,std::string(basenode_)+node);  // write out the row
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
	      writeFinite_(corr_,std::string(basenode_)+node);  // write out the diagonal
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
	   writeFinite_(corr_,std::string(basenode_)+node);  // write out the column
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
	   writeFinite_(corr_,std::string(basenode_)+node);  // write out the row
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
	      writeFinite_(corr_,std::string(basenode_)+node);  // write out the diagonal
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
	writeFinite_(corr_,std::string(basenode_)+nodeD);  // write out the diagonal
	measureRow(1);
	writeFinite_(corr_,std::string(basenode_)+nodeR);  // write out the row
	measureColumn(1);
	writeFinite_(corr_,std::string(basenode_)+nodeC);  // write out the column
     
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
        writeFinite_(corr_,std::string(basenode_)+nodeD);  // write out the diagonal
        measureRow(1);
        writeFinite_(corr_,std::string(basenode_)+nodeR);  // write out the row
        measureColumn(1);
        writeFinite_(corr_,std::string(basenode_)+nodeC);  // write out the column

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

//! Prepare the projection used by measureK, before anything is written to the file:
//!
//! Mode. With non-empty params["wallSources"] measureK projects onto the wall sources
//! ("wallSources", ProjectionMode 1); otherwise onto eigenvectors of the hopping matrix K
//! ("hoppingEigenbasis", ProjectionMode 2). wallSources and eigenvalueIndices together are an
//! error. The mode is stored as <basenode>/ProjectionMode (int, attribute "mode"). A file is never
//! mixed: if it holds a different ProjectionMode, or (files without the marker) an Eigenbasis
//! group while wall sources are requested, or propagator nodes (NonInteracting or any
//! configuration) without an Eigenbasis group while the eigenbasis is requested, measureK stops
//! with an error before writing.
//!
//! Eigenbasis selection (rule version 4). K v_n = lambda_n v_n (eigh: ascending lambda, index n
//! as in numpy.linalg.eigh). Single-particle energies eps_n = -(lambda_n + mu): the propagator
//! carries e^{delta K} e^{delta mu}, so <c(tau) c^+(0)> = e^{-eps tau} / (1 + e^{-beta eps}) and a
//! level is occupied for eps < 0. Levels = groups of eps within tol = 1e-8*max(1,max|eps|)
//! (chained over consecutive sorted values); energy index i = position in ascending eps, ties by n.
//! The eigenvectors do not depend on mu, only the selection does.
//! Non-interacting filling at mu: HOMO = highest level with eps < 0, LUMO = lowest level with
//! eps > 0; a level with |eps| <= tol is ambiguous and taken as the HOMO (warning).
//!  - default (Fermi window): the levels HOMO-below .. LUMO+above (params fermiWindowBelow = 1,
//!    fermiWindowAbove = 0 by default), each with its full degenerate subspace; levels beyond the
//!    edges of the spectrum are skipped with a warning.
//!  - params["eigenvalueIndices"] (energy indices): exactly these vectors; duplicates or indices
//!    outside [0,Nx) are an error; a listed level whose degenerate partners are not all listed
//!    gives a warning (basis-dependent selection), the partners are not added. An empty list
//!    counts as absent, i.e. selects the default (provenance identical to a default run).
//! Layout: ONE block whose m bands are the selected vectors in ascending eps (ties by n), without
//! conjugate copies: measureK writes only the key "0-0" with leaves (Nt, m, m).
//!
//! Provenance (written once, then reused): <basenode>/Eigenbasis/ (group attribute
//! conjugateCopies = 0)
//!   ruleVersion      ()        4 (attribute "rule")
//!   mu               ()        chemical potential of the selection
//!   selectionMode    ()        1 = Fermi window (default), 2 = eigenvalueIndices (attribute "mode")
//!   fermiWindowBelow, fermiWindowAbove ()   window parameters (Fermi-window mode only)
//!   requestedIndices (r)       the requested energy indices (eigenvalueIndices mode only)
//!   lambda, energy   (Nx)      lambda_n and eps_n, both in eigh order n
//!   energyOrder      (Nx)      eigh index n of energy index i
//!   indices          (m)       eigh index n of band j
//!   energyIndices    (m)       energy index of band j
//!   eigenvalues      (m)       lambda of band j
//!   energies         (m)       eps of band j
//!   roles            (m)       non-interacting role of band j ("HOMO-1", "HOMO", "LUMO", "LUMO+1", ...)
//!   eigenvectors     (Nx,m)    column j = vector of band j (sign/phase as used)
//! A stored eigenbasis is reused only if rule version, mu, selection mode, window parameters,
//! requested indices and selected eigh indices are equal and its eigenvectors are orthonormal
//! eigenvectors of K with the stored eigenvalues; otherwise measureK stops with an error before
//! writing (this includes eigenbases written by older rule versions).
template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::setupProjection_(){
    if (projectionReady_) { return; }

    const bool wallMode = hasWallSources_(this->params_);
    // an eigenvalueIndices tensor without elements (any shape) counts as absent (default
    // selection), like empty wall sources; otherwise all its elements are the requested indices
    const bool idxMode = this->params_.contains("eigenvalueIndices")
        && NSL::Tensor<double>(this->params_["eigenvalueIndices"]).numel() > 0;
    const int mode = wallMode ? 1 : 2;
    const std::string modeName = wallMode ? "wallSources" : "hoppingEigenbasis";
    const std::string root = std::string(basenode_);
    const std::string base = root + "/Eigenbasis";
    const int ruleVersion = 4;
    const std::string ruleText = "eps_n = -(lambda_n + mu); default (Fermi window): levels HOMO-below .. LUMO+above of the non-interacting filling at mu (|eps|<=tol level = HOMO), full degenerate subspaces; eigenvalueIndices: the listed ascending-eps indices; one block, bands by ascending eps, no conjugate copies";
    const int selMode = idxMode ? 2 : 1;
    const std::string selName = idxMode ? "eigenvalueIndices" : "fermiWindow";

    auto fail = [&](const std::string & msg){
        NSL::Logger::error("FermionPropagatorSpinBasis::measureK: {}", msg);
        throw std::runtime_error("FermionPropagatorSpinBasis::measureK: " + msg);
    };

    if (wallMode && idxMode) {
        fail("both wallSources and eigenvalueIndices are given; choose one projection");
    }

    // Fermi-window parameters (default below = 1, above = 0)
    auto windowParam = [&](const std::string & key, int def){
        if (!this->params_.contains(key)) { return def; }
        const int v = this->params_[key].template to<int>();
        if (v < 0) { fail(fmt::format("{} must be a non-negative integer (got {})", key, v)); }
        return v;
    };
    const int below = windowParam("fermiWindowBelow", 1);
    const int above = windowParam("fermiWindowAbove", 0);

    // ---- eigenbasis selection (no file access yet)
    const NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();
    const double mu = this->params_.contains("mu") ? this->params_["mu"].template to<double>() : 0.;
    torch::Tensor K, V;
    std::vector<double> lambda, eps;
    std::vector<NSL::size_t> order, epsIndex, sel, cluster;
    std::vector<std::string> roles;
    std::vector<int64_t> requested;
    double tolDeg = 0.;
    if (!wallMode) {
        // eigen decomposition of K on the CPU; real symmetric K gives real eigenvectors
        K = static_cast<torch::Tensor>(this->hfm_.Lat.hopping_matrix()).to(torch::kCPU).to(torch::kComplexDouble);
        torch::Tensor evals, evecs;
        if (torch::imag(K).abs().max().template item<double>() == 0.) {
            std::tie(evals, evecs) = torch::linalg_eigh(torch::real(K).contiguous(), "L");
            evecs = evecs.to(torch::kComplexDouble);
        } else {
            std::tie(evals, evecs) = torch::linalg_eigh(K, "L");
        }
        evals = evals.to(torch::kDouble).contiguous();
        lambda.assign(evals.data_ptr<double>(), evals.data_ptr<double>()+Nx);
        eps.resize(Nx);
        double maxAbs = 0.;
        for (NSL::size_t n = 0; n < Nx; n++) { eps[n] = -(lambda[n] + mu); maxAbs = std::max(maxAbs, std::abs(eps[n])); }
        tolDeg = 1e-8 * std::max(1., maxAbs);

        // levels: eps within tolDeg (chained over consecutive sorted values); ascending-energy order, ties by n
        std::vector<NSL::size_t> byEps(Nx);
        cluster.resize(Nx);
        std::iota(byEps.begin(), byEps.end(), 0);
        std::stable_sort(byEps.begin(), byEps.end(), [&](NSL::size_t a, NSL::size_t b){ return eps[a] < eps[b]; });
        NSL::size_t nLevels = 0;
        for (NSL::size_t i = 0; i < Nx; i++) {
            if (i > 0 && eps[byEps[i]] - eps[byEps[i-1]] > tolDeg) { nLevels++; }
            cluster[byEps[i]] = nLevels;
        }
        nLevels++;
        order.resize(Nx); epsIndex.resize(Nx);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](NSL::size_t a, NSL::size_t b){
            return std::make_pair(cluster[a], a) < std::make_pair(cluster[b], b);
        });
        for (NSL::size_t i = 0; i < Nx; i++) { epsIndex[order[i]] = i; }

        // non-interacting filling: HOMO level h (-1 if none), LUMO level l (nLevels if none)
        long h = -1;
        long zeroLevel = -1;
        for (NSL::size_t n = 0; n < Nx; n++) {
            if (std::abs(eps[n]) <= tolDeg) { zeroLevel = (long) cluster[n]; }
            else if (eps[n] < 0) { h = std::max(h, (long) cluster[n]); }
        }
        if (zeroLevel >= 0) {
            NSL::Logger::warn("Hopping eigenbasis: the level at eps = 0 (|eps| <= {:.1e}) is ambiguous at mu = {}; it is taken as the non-interacting HOMO", tolDeg, mu);
            h = zeroLevel;
        }
        const long l = h + 1;   // levels are ordered by eps, so the LUMO follows the HOMO
        auto role = [&](long c){
            if (c <= h) { return c == h ? std::string("HOMO") : fmt::format("HOMO-{}", h - c); }
            return c == l ? std::string("LUMO") : fmt::format("LUMO+{}", c - l);
        };

        std::set<NSL::size_t> selSet;
        if (idxMode) {
            torch::Tensor req = static_cast<torch::Tensor>(NSL::Tensor<double>(this->params_["eigenvalueIndices"])).to(torch::kCPU).to(torch::kDouble).contiguous();
            std::vector<double> reqD(req.data_ptr<double>(), req.data_ptr<double>()+req.numel());
            for (double r: reqD) {
                if (r != std::floor(r) || r < 0 || r >= (double) Nx) {
                    fail(fmt::format("eigenvalueIndices: {} is not an integer energy index in [0,{})", r, Nx));
                }
                if (!selSet.insert(order[(NSL::size_t) r]).second) {
                    fail(fmt::format("eigenvalueIndices: index {} is listed twice", r));
                }
                requested.push_back((int64_t) r);
            }
            for (NSL::size_t n: selSet) {
                std::string missing;
                for (NSL::size_t m = 0; m < Nx; m++) {
                    if (!selSet.contains(m) && cluster[m] == cluster[n]) { missing += fmt::format(" {}", epsIndex[m]); }
                }
                if (!missing.empty()) {
                    NSL::Logger::warn("Hopping eigenbasis: energy index {} (eps={:.6f}) is degenerate with the unlisted energy indices{}; the selection depends on the eigensolver's basis of that subspace", epsIndex[n], eps[n], missing);
                }
            }
        } else {
            // Fermi window: levels h-below .. l+above, clipped to the spectrum
            const long first = h - below, last = l + above;
            std::string skipped;
            for (long c = first; c <= last; c++) {
                if (c < 0 || c >= (long) nLevels) { skipped += " " + role(c); continue; }
                for (NSL::size_t n = 0; n < Nx; n++) { if ((long) cluster[n] == c) { selSet.insert(n); } }
            }
            if (!skipped.empty()) {
                NSL::Logger::warn("Hopping eigenbasis: Fermi window levels{} lie beyond the spectrum at mu = {} and are skipped", skipped, mu);
            }
        }
        sel.assign(selSet.begin(), selSet.end());
        std::sort(sel.begin(), sel.end(), [&](NSL::size_t a, NSL::size_t b){ return epsIndex[a] < epsIndex[b]; });
        if (sel.empty()) { fail("the eigenbasis selection is empty"); }   // never write an empty projection
        for (NSL::size_t n: sel) { roles.push_back(role((long) cluster[n])); }

        std::string desc;
        for (NSL::size_t j = 0; j < sel.size(); j++) {
            desc += fmt::format("{}band {}: n={} energy index {} lambda={:.6f} eps={:.6f} {}", j ? "; " : "", j, sel[j], epsIndex[sel[j]], lambda[sel[j]], eps[sel[j]], roles[j]);
        }
        if (idxMode) {
            NSL::Logger::info("Hopping eigenbasis (eigenvalueIndices selection, mu = {}): {} of {} eigenvectors, one block: {}", mu, sel.size(), Nx, desc);
        } else {
            NSL::Logger::info("Hopping eigenbasis (Fermi window HOMO-{} .. LUMO+{}, mu = {}): {} of {} eigenvectors, one block: {}", below, above, mu, sel.size(), Nx, desc);
        }

        V = evecs.index_select(1, torch::tensor(std::vector<int64_t>(sel.begin(), sel.end()), torch::kLong)).contiguous();
    }

    // ---- mode check against the file (no writes yet)
    if (this->h5_.exist(root+"/ProjectionMode")) {
        int stored = 0;
        this->h5_.read(stored, root+"/ProjectionMode");
        if (stored != mode) {
            fail(fmt::format("the file already holds propagators with ProjectionMode {} ({}), this run would use {} ({}); refusing to mix projections in one file",
                stored, stored == 1 ? "wallSources" : "hoppingEigenbasis", mode, modeName));
        }
    } else {
        // files written before the marker existed
        const bool hasEigenbasis = this->h5_.exist(base);
        bool hasPropagators = this->h5_.exist(root+"/NonInteracting/propagator");
        if (!wallMode && !hasPropagators && this->h5_.exist(root+"/markovChain")) {
            for (const auto & cfg: this->h5_.getFile().getGroup(root+"/markovChain").listObjectNames()) {
                if (this->h5_.exist(root+"/markovChain/"+cfg+"/propagator")) { hasPropagators = true; break; }
            }
        }
        if (wallMode && hasEigenbasis) {
            fail("the file holds hopping-eigenbasis propagators (Eigenbasis group) but wallSources are given; refusing to mix projections in one file");
        }
        if (!wallMode && hasPropagators && !hasEigenbasis) {
            fail("the file holds propagators without an Eigenbasis group (wall-source projection) but no wallSources are given; refusing to mix projections in one file");
        }
    }

    // ---- eigenbasis provenance: reuse (after checking) or write
    if (!wallMode) {
        const NSL::size_t nSel = sel.size();
        std::vector<double> lamSel, epsSel; std::vector<int64_t> nSelV, iSelV, orderV(order.begin(), order.end());
        for (NSL::size_t n: sel) { lamSel.push_back(lambda[n]); epsSel.push_back(eps[n]); nSelV.push_back(n); iSelV.push_back(epsIndex[n]); }
        auto asLong = [&](const std::string & node){
            NSL::Tensor<int> t; this->h5_.read(t, node);
            torch::Tensor tt = static_cast<torch::Tensor>(t).to(torch::kCPU).to(torch::kLong).contiguous();
            return std::vector<int64_t>(tt.data_ptr<int64_t>(), tt.data_ptr<int64_t>()+tt.numel());
        };
        if (this->h5_.exist(base)) {
            std::string why;
            int storedRule = 0, storedSel = 0; double storedMu = 0.;
            if (!this->h5_.exist(base+"/ruleVersion")) { why = "no ruleVersion (written by an older selection rule)"; }
            else {
                this->h5_.read(storedRule, base+"/ruleVersion");
                if (storedRule != ruleVersion) { why = fmt::format("rule version {} (this run: {})", storedRule, ruleVersion); }
            }
            if (why.empty()) {
                this->h5_.read(storedMu, base+"/mu");
                this->h5_.read(storedSel, base+"/selectionMode");
                if (std::abs(storedMu - mu) > 1e-12 * std::max(1., std::abs(mu))) { why = fmt::format("mu = {} (this run: {})", storedMu, mu); }
                else if (storedSel != selMode) { why = fmt::format("selection mode {} (this run: {})", storedSel == 2 ? "eigenvalueIndices" : "fermiWindow", selName); }
                else if (idxMode && asLong(base+"/requestedIndices") != requested) { why = "different eigenvalueIndices"; }
                else if (!idxMode) {
                    int storedBelow = -1, storedAbove = -1;
                    this->h5_.read(storedBelow, base+"/fermiWindowBelow");
                    this->h5_.read(storedAbove, base+"/fermiWindowAbove");
                    if (storedBelow != below || storedAbove != above) {
                        why = fmt::format("Fermi window HOMO-{} .. LUMO+{} (this run: HOMO-{} .. LUMO+{})", storedBelow, storedAbove, below, above);
                    }
                }
                if (why.empty() && asLong(base+"/indices") != nSelV) { why = "different selected eigenvectors"; }
            }
            if (why.empty()) {
                NSL::Tensor<Type> Vstored; NSL::Tensor<double> lamStored;
                this->h5_.read(Vstored, base+"/eigenvectors");
                this->h5_.read(lamStored, base+"/eigenvalues");
                torch::Tensor Vs = static_cast<torch::Tensor>(Vstored).to(torch::kCPU).to(torch::kComplexDouble);
                torch::Tensor ls = static_cast<torch::Tensor>(lamStored).to(torch::kCPU).to(torch::kDouble);
                torch::Tensor lc = torch::tensor(lamSel, torch::kDouble);
                bool ok = Vs.dim() == 2 && Vs.size(0) == (int64_t) Nx && Vs.size(1) == (int64_t) nSel && ls.numel() == (int64_t) nSel;
                ok = ok && (ls - lc).abs().max().template item<double>() <= tolDeg
                        && (torch::matmul(K, Vs) - Vs * ls.to(torch::kComplexDouble)).abs().max().template item<double>() <= tolDeg
                        && (torch::matmul(Vs.conj().transpose(0,1), Vs) - torch::eye(nSel, torch::kComplexDouble)).abs().max().template item<double>() <= 1e-8;
                if (ok) { V = Vs; } else { why = "stored eigenvectors/eigenvalues do not match K"; }
            }
            if (!why.empty()) {
                fail(fmt::format("the hopping eigenbasis stored in {} differs from this run: {}; refusing to mix selections in one file", base, why));
            }
            NSL::Logger::info("Hopping eigenbasis: using the eigenvectors stored in {}", base);
        } else {
            auto attr = [&](const std::string & node, const std::string & name, const std::string & value){
                HighFive::DataSet ds = this->h5_.getFile().getDataSet(node);
                ds.createAttribute<std::string>(name, HighFive::DataSpace::From(value)).write(value);
            };
            auto dbl = [](const std::vector<double> & v){ return NSL::Tensor<double>(torch::tensor(v, torch::kDouble)); };
            auto itg = [](const std::vector<int64_t> & v){ return NSL::Tensor<int>(torch::tensor(v, torch::kLong).to(torch::kInt)); };
            NSL::Tensor<double> lamAll = dbl(lambda), epsAll = dbl(eps), lamOut = dbl(lamSel), epsOut = dbl(epsSel);
            NSL::Tensor<int> orderOut = itg(orderV), idxOut = itg(nSelV), eidxOut = itg(iSelV);
            NSL::Tensor<Type> Vout(V.clone());
            this->h5_.write(ruleVersion, base+"/ruleVersion");  attr(base+"/ruleVersion", "rule", ruleText);
            const int conjugateCopies = 0;
            this->h5_.getFile().getGroup(base).createAttribute<int>("conjugateCopies", HighFive::DataSpace::From(conjugateCopies)).write(conjugateCopies);
            this->h5_.write(mu, base+"/mu");
            this->h5_.write(selMode, base+"/selectionMode");    attr(base+"/selectionMode", "mode", selName);
            if (idxMode) { NSL::Tensor<int> reqOut = itg(requested); this->h5_.write(reqOut, base+"/requestedIndices"); }
            else { this->h5_.write(below, base+"/fermiWindowBelow"); this->h5_.write(above, base+"/fermiWindowAbove"); }
            this->h5_.write(lamAll,   base+"/lambda");
            this->h5_.write(epsAll,   base+"/energy");
            this->h5_.write(orderOut, base+"/energyOrder");
            this->h5_.write(idxOut,   base+"/indices");
            this->h5_.write(eidxOut,  base+"/energyIndices");
            this->h5_.write(lamOut,   base+"/eigenvalues");
            this->h5_.write(epsOut,   base+"/energies");
            this->h5_.getFile().createDataSet<std::string>(base+"/roles", HighFive::DataSpace::From(roles)).write(roles);
            this->h5_.write(Vout,     base+"/eigenvectors");
        }

        // one block: rows of eigSources_ (m x Nx) are the band vectors (columns of V)
        const NSL::Device device = this->params_["device"].template to<NSL::Device>();
        eigSources_ = NSL::Tensor<Type>(V.transpose(0,1).contiguous().to(device.device().device()));
        corrK_.resize(this->params_["Nt"].template to<NSL::size_t>(), nSel, nSel);
    }

    // ---- mode marker
    if (!this->h5_.exist(root+"/ProjectionMode")) {
        this->h5_.write(mode, root+"/ProjectionMode");
        HighFive::DataSet modeSet = this->h5_.getFile().getDataSet(root+"/ProjectionMode");
        modeSet.createAttribute<std::string>("mode", HighFive::DataSpace::From(modeName)).write(modeName);
        this->h5_.getFile().flush();
    }

    projectionReady_ = true;
} // setupProjection_()

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
    requireFinite_(proj, std::string(basenode_)+node);   // nothing of this node is written otherwise

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
    setupProjection_();

    // Reset memory
    // - Result correlator
    corr_ = Type(0);
    corrK_ = Type(0);

    std::string node;
    std::string prefix;

    bool PiSigma;

    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();

    NSL::size_t tsrcStep = ceil((Nt+0.0)/NumberTimeSources);

    // projection sources as rows, (kDim*bands) x Nx, row k*bands+sigma = source (k,sigma)
    NSL::Tensor<Type> wallSources;
    if (wallMode) {
        // wall sources (blocks x bands x Nx): even k = source, odd k = conj(source)
        NSL::Tensor<Type> blocks(params_["wallSources"]);
        NSL::Tensor<Type> doubled(params_["device"].to<NSL::Device>(), 2 * blocks.shape(0), blocks.shape(1), blocks.shape(2));
        doubled(NSL::Slice(0,NSL::none_t(), 2), NSL::Ellipsis()) = blocks;
        doubled(NSL::Slice(1,NSL::none_t(), 2), NSL::Ellipsis()) = NSL::LinAlg::conj(blocks);
        doubled.reshape(doubled.shape(0)*doubled.shape(1), doubled.shape(2));
        wallSources = doubled;
    } else {
        // hopping eigenbasis: one block (kDim = 1) whose m bands are the selected vectors,
        // no conjugate copies -> only the key "0-0", leaves (Nt, m, m)
        wallSources = eigSources_;
    }

    // if non-interacting (phi = 0) write to /NonInteracting, otherwise to /markovChain/cfgID
    std::string measPrefix;
    if ((phi_ == Type(0)).all()) {
        prefix = "/NonInteracting/propagator";
        measPrefix = "/NonInteracting/measurements";
    } else {
        prefix = fmt::format("/markovChain/{}/propagator",cfgID);
        measPrefix = fmt::format("/markovChain/{}/measurements",cfgID);
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

                // charge Q = Tr_x G(0,0) of the full (unprojected) equal-time propagator
                node = fmt::format("{}/{}/Q",measPrefix,spinNames[s]);
                if (!skip_(this->params_["overwrite"],node)) {
                    if (PiSigma) {
                        hfm_.populate(phi_,spin_);
                        calcPiSigma(tsrc);
                        PiSigma = false;
                    }
                    Type Q = traceG00_();
                    if (!std::isfinite(static_cast<double>(Q.real())) || !std::isfinite(static_cast<double>(Q.imag()))) {
                        NSL::Logger::error("FermionPropagatorSpinBasis: non-finite charge in {}; nothing written for this node", std::string(basenode_)+node);
                        throw std::runtime_error("FermionPropagatorSpinBasis: non-finite charge in " + std::string(basenode_)+node);
                    }
                    this->h5_.write(Q, std::string(basenode_)+node);
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
    }
    setupProjection_();

    // write the momenta out
    if (!h5_.exist(std::string(basenode_)+"/Momenta")){
        NSL::Tensor<double> momenta = params_["momenta"];
        this->h5_.write(momenta,std::string(basenode_)+"/Momenta");
    }

    // write the non interacting correlator 
    std::string node = "/NonInteracting/propagator";
    // this is a shortcut, we don't need to calculate the non-interacting 
    // correlators if we won't update the file
    if(!skip_(this->params_["overwrite"],node)
        || !skip_(this->params_["overwrite"],"/NonInteracting/measurements/spinUp/Q")
        || !skip_(this->params_["overwrite"],"/NonInteracting/measurements/spinDown/Q")) {
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
Type FermionPropagatorSpinBasis<Type,LatticeType,FermionMatrixType>::traceG00_(){
    // the t = 0 term of measureDiagonal (identical operations), traced over the sites
    std::tie( Qnew_, Dnew_, Vnew_ ) = udtHost_( NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(0,NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(SIGMA_D_(0,NSL::Ellipsis())));
    NSL::Tensor<Type> Ut = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnew_));
    NSL::Tensor<Type> Tt = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));
    torch::Tensor G00 = static_cast<torch::Tensor>(NSL::LinAlg::mat_mul(Ut, Tt));
    return G00.diagonal(0, -2, -1).sum().to(torch::kCPU).template item<Type>();
} // traceG00_()

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
    std::tie( Qnew_, Dnew_, Vnew_ ) = udtHost_( NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(0,NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(SIGMA_D_(0,NSL::Ellipsis())));

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
	   std::tie( uut_,ddt_,vvt_ ) = udtHost_(vut_);
	}


    // std::cout << "gets here 2" << std::endl;
    Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()),uut_(NSL::Slice(1,Nt),NSL::Ellipsis()));
    Fk_Dt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = ddt_(NSL::Slice(1,Nt),NSL::Ellipsis());
	Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vvt_(NSL::Slice(1,Nt),NSL::Ellipsis()),SIGMA_V_(NSL::Slice(1,Nt),NSL::Ellipsis()));

    // std::cout << "gets here 3" << std::endl;
    std::tie( Qnewtt_, Dnewtt_, Vnewtt_ ) = udtHost_( NSL::LinAlg::solve(Fk_Vt_(NSL::Slice(1,Nt),NSL::Ellipsis()), NSL::LinAlg::adjoint(Fk_Ut_(NSL::Slice(1,Nt),NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(Fk_Dt_(NSL::Slice(1,Nt),NSL::Ellipsis())));

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
    std::tie( Qnew_, Dnew_, Vnew_ ) = udtHost_( NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(0,NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(SIGMA_D_(0,NSL::Ellipsis())));

    invAp1F_Ut_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnew_));
    invAp1F_Tt_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));

    corr_(tsrc,NSL::Ellipsis()) = NSL::LinAlg::mat_mul( invAp1F_Ut_(0,NSL::Ellipsis()),invAp1F_Tt_(0,NSL::Ellipsis()) );

    // t>0 terms
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::adjoint(PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),NSL::LinAlg::solve(SIGMA_V_(NSL::Slice(1,Nt),NSL::Ellipsis()),NSL::LinAlg::diag_embed(1./SIGMA_D_(NSL::Slice(1,Nt),NSL::Ellipsis()))));
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) + NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(NSL::LinAlg::diag_embed(PI_D_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),PI_V_(NSL::Slice(0,Nt-1),NSL::Ellipsis())),SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis()));

    if (!this->hfm_.stabilityMethod.compare("SVD")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::svd(vut_);
    } else { //(!this->hfm_.stabilityMethod.compare("QR")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = udtHost_(vut_);
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
	   std::tie( Qnewt_, Dnewt_, Vnewt_ ) = udtHost_(vut_);
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
    std::tie( Qnew_, Dnew_, Vnew_ ) = udtHost_( NSL::LinAlg::solve(SIGMA_V_(0,NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(0,NSL::Ellipsis())),false ) + NSL::LinAlg::diag_embed(SIGMA_D_(0,NSL::Ellipsis())));

    invAp1F_Ut_(0,NSL::Ellipsis()) = NSL::LinAlg::solve(NSL::LinAlg::mat_mul(Vnew_, SIGMA_V_(0,NSL::Ellipsis())) , NSL::LinAlg::diag_embed(1./Dnew_));
    invAp1F_Tt_(0,NSL::Ellipsis()) = NSL::LinAlg::adjoint(NSL::LinAlg::mat_mul( SIGMA_U_(0,NSL::Ellipsis()), Qnew_ ));

    corr_(tsrc,NSL::Ellipsis()) = NSL::LinAlg::mat_mul( invAp1F_Ut_(0,NSL::Ellipsis()),invAp1F_Tt_(0,NSL::Ellipsis()) );

    // t>0
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = NSL::LinAlg::mat_mul(NSL::LinAlg::solve(PI_V_(NSL::Slice(0,Nt-1),NSL::Ellipsis()), NSL::LinAlg::adjoint(SIGMA_U_(NSL::Slice(1,Nt),NSL::Ellipsis())), false ),NSL::LinAlg::diag_embed(1./PI_D_(NSL::Slice(0,Nt-1),NSL::Ellipsis())));
    vut_(NSL::Slice(1,Nt),NSL::Ellipsis()) = vut_(NSL::Slice(1,Nt),NSL::Ellipsis())+ NSL::LinAlg::mat_mul(NSL::LinAlg::mat_mul(NSL::LinAlg::diag_embed(SIGMA_D_(NSL::Slice(1,Nt),NSL::Ellipsis())),SIGMA_V_(NSL::Slice(1,Nt),NSL::Ellipsis())),PI_U_(NSL::Slice(0,Nt-1),NSL::Ellipsis()));

    if (!this->hfm_.stabilityMethod.compare("SVD")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = NSL::LinAlg::svd(vut_);
    } else { //(!this->hfm_.stabilityMethod.compare("QR")) {
           std::tie( Qnewt_, Dnewt_, Vnewt_ ) = udtHost_(vut_);
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
	   std::tie( Qnewt_, Dnewt_, Vnewt_ ) = udtHost_(vut_);
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
      	  std::tie( uu_,dd_,vv_ ) = udtHost_(vu_);
      	  PI_U_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(PI_U_(t,NSL::Ellipsis()),uu_);
	  PI_D_(t,NSL::Ellipsis()) = dd_;
	  PI_V_(t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,PI_V_(t-1,NSL::Ellipsis()));

	  // SIGMA(t)=Fk(Nt-1)...Fk(t)
	  vu_ = NSL::LinAlg::mat_mul(SIGMA_V_(Nt-t,NSL::Ellipsis()),SIGMA_U_(Nt-1-t,NSL::Ellipsis()));
	  vu_ = NSL::LinAlg::mat_mul(vu_,NSL::LinAlg::diag(SIGMA_D_(Nt-1-t,NSL::Ellipsis())));
	  vu_ = NSL::LinAlg::mat_mul(NSL::LinAlg::diag(SIGMA_D_(Nt-t,NSL::Ellipsis())),vu_);
	  std::tie( uu_,dd_,vv_ ) = udtHost_(vu_);
	  SIGMA_U_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(SIGMA_U_(Nt-t,NSL::Ellipsis()),uu_);
	  SIGMA_D_(Nt-1-t,NSL::Ellipsis()) = dd_;
	  SIGMA_V_(Nt-1-t,NSL::Ellipsis()) = NSL::LinAlg::mat_mul(vv_,SIGMA_V_(Nt-1-t,NSL::Ellipsis()));
      }
  }

} // calcPiSigma(tsrc)


} // namespace NSL::Measure::Hubbard

#endif // NSL_FERMION_PROPAGATOR_SPIN_BASIS_TPP
