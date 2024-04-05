#ifndef NSL_TWO_BODY_CORRELATION_FUNCTION_TPP
#define NSL_TWO_BODY_CORRELATION_FUNCTION_TPP

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
class TwoBodyCorrelator: public Measurement {
    public:
        TwoBodyCorrelator(LatticeType & lattice, NSL::Parameter params, NSL::H5IO & h5, std::string basenode_):
            Measurement(params, h5),
            hfm_(lattice, params),
            cg_(hfm_, NSL::FermionMatrix::MMdagger),
            corrK_(
                params["device"].to<NSL::Device>(),
                params["wallSources"].shape(1).to<NSL::size_t>(),
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
                ),
            corrKPool_(
                params["device"].to<NSL::Device>(),
                params["wallSources"].shape(0).to<NSL::size_t>(),
                params["wallSources"].shape(0).to<NSL::size_t>(),
                // NSL::size_t (species_.size()), // I want to have a size 2 in this place
                params["Nt"].to<NSL::size_t>(),
                params["wallSources"].shape(1).to<NSL::size_t>(),
                params["wallSources"].shape(1).to<NSL::size_t>()
                ),
	        srcVecK_(
                params["device"].to<NSL::Device>(),
                params["wallSources"].shape(1).to<NSL::size_t>(),
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
            ),
            phi_(
                params["device"].to<NSL::Device>(),
                params["Nt"].to<NSL::size_t>(),
                params["Nx"].to<NSL::size_t>()
            ),
            basenode_(basenode_)
        {}

        TwoBodyCorrelator(LatticeType & lattice, NSL::Parameter params,NSL::H5IO & h5):
            TwoBodyCorrelator(
                lattice,
                params,
                h5, 
                params["name"]
            )
        {}

        //! Calculate the \f( N_t \times N_x \times N_x \f) correlators, i.e. 
        //! Propagators with averaged second time coordinate
        void measure() override;

        void measure(NSL::size_t NumberTimeSources);

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

        bool eq_modBZ_(const NSL::Tensor<double>& k, const NSL::Tensor<double>& q, double eps = 1e-15) {
            // std::cout << "WE ARE HERE" << std::endl;
            NSL::Tensor<double> diff = k-q;
            // These two vectors are linearly independent vectors that take us to
            // Gamma points in neighboring cells in the reciprocal lattice.
            std::vector<double> Cv = {4 * M_PI / 3, 0};
            std::vector<double> Dv = {-2 * M_PI /3, 2 * M_PI / sqrt(3.0)};

            NSL::Tensor<double> C(Cv.size());
            C = Cv;
            NSL::Tensor<double> D(Dv.size());
            D = Dv;

            // Two vectors are equal (mod BZ) if their difference is 0 (mod BZ).
            // In practice, that means an integer number of 'jumps' between them.
            // In other words, if
            //     k = a  C + b  D + remainder      with integers a , b
            //     q = a' C + b' D + remainder'     with integers a', b'
            // Then k = q (mod BZ) if remainder = remainder'.
            // Subtracting,
            // 0 = (k-q) = (a-a') C + (b-b') D + (remainder-remainder')
            //           = m C + n D + ZERO
            //           = m C + n D with integers m, n.
            // So, we can solve
            // diff = m C + n D for m,n
            // and if they are integers, k and q are equal (mod BZ).
            double denominator = C(1) * D(0) - C(0) * D(1);
            double m = (diff(1) * D(0) - diff(0) * D(1)) / denominator;
            double n = (diff(0) * C(1) - diff(1) * C(0)) / denominator;
            
            return (abs(round(m)- m) < eps && abs(round(n)- n) < eps);
            // if (abs(round(m)- m) < eps && abs(round(n)- n) < eps) return true;

            // return false;
        }

        // NSL::Tensor<Type> kroneckerProduct_(const NSL::Tensor<Type>& A, const NSL::Tensor<Type>& B) {
            
        //     int rowsA = A.shape(1);
        //     int colsA = A.shape(2);
        //     int rowsB = B.shape(1);
        //     int colsB = B.shape(2);

        //     // Result Tensor has shape ( (rowsA*rowsB) x (colsA*colsB) )
        //     NSL::Tensor<Type> result(params_["device"].to<NSL::Device>(), A.shape(0), rowsA * rowsB, colsA * colsB);

        //     for (int i = 0; i < rowsA; ++i) {
        //         for (int j = 0; j < colsA; ++j) {
        //             for (int k = 0; k < rowsB; ++k) {
        //                 for (int l = 0; l < colsB; ++l) {
        //                     result(NSL::Slice(), i * rowsB + k, j * colsB + l) = A(NSL::Slice(),i,j) * B(NSL::Slice(),k,l);
        //                 }
        //             }
        //         }
        //     }
        //     return result;
        // }

        FermionMatrixType hfm_;
        NSL::LinAlg::CG<Type> cg_;

        NSL::Tensor<Type> corrK_;
        NSL::Tensor<Type> corrKPool_;
        std::unordered_map<NSL::Hubbard::Species, NSL::Tensor<Type>> corrPool_;
        NSL::Tensor<Type> srcVecK_;

        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Iz1Sz1_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Iz1Sz0_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Iz1Szn1_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Iz0Sz1_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Iz0Szn1_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Izn1Sz1_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Izn1Sz0_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Izn1Szn1_;

        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI0S1Iz0Szn1_;

        NSL::Tensor<Type> phi_;

        std::string basenode_;

        void I1S1Iz1Sz1_();
        void I1S1Iz1Sz0_();
        void I1S1Iz1Szn1_();
        void I1S1Iz0Sz1_();
        void I1S1Iz0Szn1_();
        void I1S1Izn1Sz1_();
        void I1S1Izn1Sz0_();
        void I1S1Izn1Szn1_();

        void I0S1Iz0Szn1_();
};


template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::measure(NSL::size_t NumberTimeSources) {
    NSL::size_t Nx = this->params_["Nx"].template to<NSL::size_t>();
    NSL::size_t Nt = this->params_["Nt"].template to<NSL::size_t>();
    
    NSL::size_t tsrcStep = Nt/NumberTimeSources;

    int kDim = params_["wallSources"].shape(0);
    int bDim = params_["wallSources"].shape(1);

    NSL::Tensor<double> momenta = params_["momenta"];
    for (int kSinkPart=0; kSinkPart<kDim; kSinkPart++) {
        for (int kSinkHole=0; kSinkHole<kDim; kSinkHole++) {
            for (int kSrcHole=0; kSrcHole<kDim; kSrcHole++) {
                for (int kSrcPart=0; kSrcPart<kDim; kSrcPart++) {
                    if (!(eq_modBZ_(momenta(kSinkPart,NSL::Slice()) + momenta(kSinkHole,NSL::Slice()), momenta(kSrcPart, NSL::Slice()) + momenta(kSrcHole,NSL::Slice())))) {
                        continue;
                    }
                    cI1S1Iz1Sz1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart] = NSL::Tensor<Type> (params_["device"].to<NSL::Device>(), Nt, bDim, bDim, bDim, bDim);
                    cI1S1Iz1Sz0_[kSinkPart][kSinkHole][kSrcHole][kSrcPart] = NSL::Tensor<Type> (params_["device"].to<NSL::Device>(), Nt, bDim, bDim, bDim, bDim);
                    cI1S1Iz1Szn1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart] = NSL::Tensor<Type> (params_["device"].to<NSL::Device>(), Nt, bDim, bDim, bDim, bDim);
                    cI1S1Iz0Sz1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart] = NSL::Tensor<Type> (params_["device"].to<NSL::Device>(), Nt, bDim, bDim, bDim, bDim);
                    // cI1S1Iz0Szn1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart] = NSL::Tensor<Type> (params_["device"].to<NSL::Device>(), Nt, bDim * bDim, bDim * bDim);
                    cI1S1Iz0Szn1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart] = NSL::Tensor<Type> (params_["device"].to<NSL::Device>(), Nt, bDim, bDim, bDim, bDim);
                    cI1S1Izn1Sz1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart] = NSL::Tensor<Type> (params_["device"].to<NSL::Device>(), Nt, bDim, bDim, bDim, bDim);
                    cI1S1Izn1Sz0_[kSinkPart][kSinkHole][kSrcHole][kSrcPart] = NSL::Tensor<Type> (params_["device"].to<NSL::Device>(), Nt, bDim, bDim, bDim, bDim);
                    cI1S1Izn1Szn1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart] = NSL::Tensor<Type> (params_["device"].to<NSL::Device>(), Nt, bDim, bDim, bDim, bDim);

                    cI0S1Iz0Szn1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart] = NSL::Tensor<Type> (params_["device"].to<NSL::Device>(), Nt, bDim, bDim, bDim, bDim);
                }
            }
        }
    }
    corrPool_[NSL::Hubbard::Particle] = NSL::Tensor<Type> (params_["device"].to<NSL::Device>(), kDim, kDim, Nt, bDim, bDim);
    corrPool_[NSL::Hubbard::Hole] = NSL::Tensor<Type> (params_["device"].to<NSL::Device>(), kDim, kDim, Nt, bDim, bDim);


    for (NSL::size_t tsrc = 0; tsrc<Nt; tsrc+=tsrcStep) {
        
        for (NSL::Hubbard::Species species : {NSL::Hubbard::Particle, NSL::Hubbard::Hole}) {
            // Reset memory
            // - Result correlator
            corrKPool_ = Type(0);

            // populate the fermion matrix using the free configuration
            hfm_.populate(phi_,species);

            for (int kSrc=0; kSrc<kDim; kSrc++ ) {
                // Define a wall source
                srcVecK_(NSL::Slice(),tsrc,NSL::Slice()) = NSL::Tensor<NSL::complex<double>> (params_["wallSources"])(kSrc,NSL::Slice(),NSL::Slice());

                // invert MM^dagger
                NSL::Tensor<Type> invMMdag = cg_(srcVecK_);

                // Reset memory
                // - Source vector
                srcVecK_ = Type(0);

                // back multiply M^dagger to obtain M^{-1}
                // invM is of shape Nx x Nt x Nx
                NSL::Tensor<Type> invM = hfm_.Mdagger(invMMdag);

                // We shift the 1st axis (time-axis) if invM by tsrc and apply anti periodic 
                // boundary conditions
                // shift t -> t - tsrc
                invM.shift( -tsrc, -2, -Type(1) );

                // Average over all source times
                // corrK_ += invM; // I changed something here!!!!!
                corrK_ = invM;

                for (int kSink=0; kSink<kDim; kSink++) {
                    for (NSL::size_t t = 0; t<Nt; t++) {
                        for (int sigmaSink=0; sigmaSink<bDim; sigmaSink++) {
                            for (int sigmaSrc=0; sigmaSrc<bDim; sigmaSrc++) {
                                corrKPool_(kSink,kSrc,t,sigmaSink,sigmaSrc) = NSL::LinAlg::inner_product( NSL::Tensor<NSL::complex<double>> (params_["wallSources"])(kSink,sigmaSink,NSL::Slice()), corrK_(sigmaSrc,t,NSL::Slice()));
                            }
                        }
                    }
                }
                // Reset memory
                // - Result correlator
                corrK_ = Type(0);
            } // for kSrc

            corrPool_[species] = corrKPool_;
        } // for species

        I1S1Iz1Sz1_();
        I1S1Iz1Sz0_();
        I1S1Iz1Szn1_();
        I1S1Iz0Sz1_();
        I1S1Iz0Szn1_();
        I1S1Izn1Sz1_();
        I1S1Izn1Sz0_();
        I1S1Izn1Szn1_();

        I0S1Iz0Szn1_();
    } // for tscr

} // measure(Ntsrc);

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::measure(){
    NSL::Logger::info("Start Measuring Hubbard::TwoPointCorrelator");

    // This is the default basenode we used so far
    // ToDo: this should go into the const

    // write the non interacting correlator 
    std::string node;
    node = "/NonInteracting/correlators/twobody";
    // this is a shortcut, we don't need to calculate the non-interacting 
    // correlators if we won't update the file

    // measure the non-interacting theory
    // U = 0 <=> phi = 0
    phi_ = Type(0);

    measure(1);
    std::string momNode;
    for ( const auto &[kSinkPart, value1]: cI1S1Iz1Sz1_ ) {
        for ( const auto &[kSinkHole, value2]: value1 ) {
            for ( const auto &[kSrcHole, value3]: value2 ) {
                for ( const auto &[kSrcPart, value4]: value3 ) {
                    // write the result
                    momNode = fmt::format("/cI1S1Iz1Sz1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                    this->h5_.write(cI1S1Iz1Sz1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                    momNode = fmt::format("/cI1S1Iz1Sz0/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                    this->h5_.write(cI1S1Iz1Sz0_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                    momNode = fmt::format("/cI1S1Iz1Szn1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                    this->h5_.write(cI1S1Iz1Szn1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                    momNode = fmt::format("/cI1S1Iz0Sz1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                    this->h5_.write(cI1S1Iz0Sz1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                    momNode = fmt::format("/cI1S1Iz0Szn1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                    this->h5_.write(cI1S1Iz0Szn1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                    momNode = fmt::format("/cI1S1Izn1Sz1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                    this->h5_.write(cI1S1Izn1Sz1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                    momNode = fmt::format("/cI1S1Izn1Sz0/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                    this->h5_.write(cI1S1Izn1Sz0_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                    momNode = fmt::format("/cI1S1Izn1Szn1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                    this->h5_.write(cI1S1Izn1Szn1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);


                    momNode = fmt::format("/cI0S1Iz0Szn1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                    this->h5_.write(cI0S1Iz0Szn1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);
                }
            }
        }
    }
    // Interacting Correlators
    // Initialize memory for the configurations
    // get the range of configuration ids from the h5file
    auto [minCfg, maxCfg] = this->h5_.getMinMaxConfigs(std::string(basenode_)+"/markovChain");
    NSL::size_t saveFreq = this->params_["save frequency"];
    
    NSL::Logger::info("Found trajectories: {} to {} with save frequency {}",
        minCfg, maxCfg, saveFreq
    );

    // Determine the number of time sources:
    for (NSL::size_t cfgID = minCfg; cfgID<=maxCfg; ++cfgID){
        // this is a shortcut, we don't need to invert if we don't overwrite
        // data

        node = fmt::format("/markovChain/{}/correlators/twobody",cfgID);

        if (skip_(this->params_["overwrite"],node)) {
            NSL::Logger::info("Config #{} already has correlators, skipping... ", cfgID);
	        continue;
	    } 

        NSL::Logger::info("Calculating Correlator on {}/{}", cfgID, maxCfg);

        // read configuration 
        this->h5_.read(phi_,fmt::format("{}/markovChain/{}/phi",std::string(basenode_),cfgID));

        // compute the correlator. The result is stored in corrPool_
        measure(this->params_["Number Time Sources"]);
        std::string momNode;
        for ( const auto &[kSinkPart, value1]: cI1S1Iz1Sz1_ ) {
            for ( const auto &[kSinkHole, value2]: value1 ) {
                for ( const auto &[kSrcHole, value3]: value2 ) {
                    for ( const auto &[kSrcPart, value4]: value3 ) {
                        // write the result
                        momNode = fmt::format("/cI1S1Iz1Sz1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                        this->h5_.write(cI1S1Iz1Sz1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Iz1Sz0/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                        this->h5_.write(cI1S1Iz1Sz0_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Iz1Szn1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                        this->h5_.write(cI1S1Iz1Szn1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Iz0Sz1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                        this->h5_.write(cI1S1Iz0Sz1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Iz0Szn1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                        this->h5_.write(cI1S1Iz0Szn1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Izn1Sz1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                        this->h5_.write(cI1S1Izn1Sz1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Izn1Sz0/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                        this->h5_.write(cI1S1Izn1Sz0_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Izn1Szn1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                        this->h5_.write(cI1S1Izn1Szn1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);


                        momNode = fmt::format("/cI0S1Iz0Szn1/{}-{}-{}-{}",kSinkPart,kSinkHole,kSrcPart,kSrcHole);
                        this->h5_.write(cI0S1Iz0Szn1_[kSinkPart][kSinkHole][kSrcHole][kSrcPart],std::string(basenode_)+node+momNode);
                    }
                }
            }
        }
        
    } // for cfgI

} // measure()

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Iz1Sz1_() {
    int kDim = params_["wallSources"].shape(0).to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).to<NSL::size_t>();
    int Nt = params_["Nt"].to<NSL::size_t>();

    std::vector<NSL::size_t> tDim{0};
    
    // NSL::Tensor<Type> eye(params_["device"].to<NSL::Device>(),kDim,kDim,Nt,bDim,bDim);
    // for (int k=0; k<kDim; k++) {
    //     for (int nt=0;nt<Nt; nt++){
    //         eye(k,k,nt,NSL::Slice(), NSL::Slice()) = NSL::Matrix::Identity<Type> (bDim);
    //     }
    // }

    for ( const auto &[kSinkPart, value1]: cI1S1Iz1Sz1_ ) {
        for ( const auto &[kSinkHole, value2]: value1 ) {
            for ( const auto &[kSrcHole, value3]: value2 ) {
                for ( const auto &[kSrcPart, value4]: value3 ) {

                    for (int bSinkPart=0; bSinkPart<bDim; bSinkPart++) {
                        for (int bSinkHole=0; bSinkHole<bDim; bSinkHole++) {
                            for (int bSrcHole=0; bSrcHole<bDim; bSrcHole++) {
                                for (int bSrcPart=0; bSrcPart<bDim; bSrcPart++) {

                                    cI1S1Iz1Sz1_[kSinkPart][kSrcPart][kSinkHole][kSrcHole](NSL::Slice(), bSinkPart, bSrcPart, bSinkHole, bSrcHole)
                                    
                                    += ((NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart), tDim) 
                                    * NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole), tDim)
                                    
                                    - NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole), tDim) 
                                    * NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart), tDim))

                                    // + eye(kSinkHole, kSrcHole, NSL::Slice(), bSinkHole, bSrcHole) * corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart)

                                    / Type(params_["Number Time Sources"]));
                                }
                            }
                        }
                    } 
                }
            }
        }
    }
}

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Iz1Sz0_() {
    int kDim = params_["wallSources"].shape(0).to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).to<NSL::size_t>();
    int Nt = params_["Nt"].to<NSL::size_t>();

    std::vector<NSL::size_t> tDim{0};

    for ( const auto &[kSinkPart, value1]: cI1S1Iz1Sz0_ ) {
        for ( const auto &[kSinkHole, value2]: value1 ) {
            for ( const auto &[kSrcHole, value3]: value2 ) {
                for ( const auto &[kSrcPart, value4]: value3 ) {

                    for (int bSinkPart=0; bSinkPart<bDim; bSinkPart++) {
                        for (int bSinkHole=0; bSinkHole<bDim; bSinkHole++) {
                            for (int bSrcHole=0; bSrcHole<bDim; bSrcHole++) {
                                for (int bSrcPart=0; bSrcPart<bDim; bSrcPart++) {

                                    cI1S1Iz1Sz0_[kSinkPart][kSrcPart][kSinkHole][kSrcHole](NSL::Slice(), bSinkPart, bSrcPart, bSinkHole, bSrcHole)
                                    
                                    += -1*(0.5 * ((NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart), tDim) 
                                    * corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole)
                                    
                                    - NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole), tDim) 
                                    * corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart)
                                    
                                    - corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole) 
                                    * NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart), tDim)
                                    
                                    + corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart) 
                                    * NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole), tDim)) 
                                    
                                    / Type(params_["Number Time Sources"])));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Iz1Szn1_() {
    int kDim = params_["wallSources"].shape(0).to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).to<NSL::size_t>();
    int Nt = params_["Nt"].to<NSL::size_t>();
    
    // NSL::Tensor<Type> eye(params_["device"].to<NSL::Device>(),kDim,kDim,Nt,bDim,bDim);
    // for (int k=0; k<kDim; k++) {
    //     for (int nt=0;nt<Nt; nt++){
    //         eye(k,k,nt,NSL::Slice(), NSL::Slice()) = NSL::Matrix::Identity<Type> (bDim);
    //     }
    // }

    for ( const auto &[kSinkPart, value1]: cI1S1Iz1Szn1_ ) {
        for ( const auto &[kSinkHole, value2]: value1 ) {
            for ( const auto &[kSrcHole, value3]: value2 ) {
                for ( const auto &[kSrcPart, value4]: value3 ) {

                    for (int bSinkPart=0; bSinkPart<bDim; bSinkPart++) {
                        for (int bSinkHole=0; bSinkHole<bDim; bSinkHole++) {
                            for (int bSrcHole=0; bSrcHole<bDim; bSrcHole++) {
                                for (int bSrcPart=0; bSrcPart<bDim; bSrcPart++) {

                                    cI1S1Iz1Szn1_[kSinkPart][kSrcPart][kSinkHole][kSrcHole](NSL::Slice(), bSinkPart, bSrcPart, bSinkHole, bSrcHole)
                                    
                                    += ((corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart) 
                                    * corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole)
                                    
                                    - corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole) 
                                    * corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart))

                                    // + eye(kSinkHole, kSrcHole, NSL::Slice(), bSinkHole, bSrcHole) * corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart)

                                    / Type(params_["Number Time Sources"]));
                                }
                            }
                        }
                    } 
                }
            }
        }
    }
}

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Iz0Sz1_() {
    int kDim = params_["wallSources"].shape(0).to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).to<NSL::size_t>();
    int Nt = params_["Nt"].to<NSL::size_t>();

    std::vector<NSL::size_t> tDim{0};

    for ( const auto &[kSinkPart, value1]: cI1S1Iz0Sz1_ ) {
        for ( const auto &[kSinkHole, value2]: value1 ) {
            for ( const auto &[kSrcHole, value3]: value2 ) {
                for ( const auto &[kSrcPart, value4]: value3 ) {

                    for (int bSinkPart=0; bSinkPart<bDim; bSinkPart++) {
                        for (int bSinkHole=0; bSinkHole<bDim; bSinkHole++) {
                            for (int bSrcHole=0; bSrcHole<bDim; bSrcHole++) {
                                for (int bSrcPart=0; bSrcPart<bDim; bSrcPart++) {

                                    cI1S1Iz0Sz1_[kSinkPart][kSrcPart][kSinkHole][kSrcHole](NSL::Slice(), bSinkPart, bSrcPart, bSinkHole, bSrcHole)
                                    
                                    += (0.5 * ((NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart), tDim)
                                    * NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole), tDim)
                                    
                                    - NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole), tDim)
                                    * NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart), tDim)
                                    
                                    - NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole), tDim) 
                                    * NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart), tDim)
                                    
                                    + NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart), tDim) 
                                    * NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole), tDim)) 
                                    
                                    / Type(params_["Number Time Sources"])));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Iz0Szn1_() {
    int kDim = params_["wallSources"].shape(0).to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).to<NSL::size_t>();
    int Nt = params_["Nt"].to<NSL::size_t>();

    for ( const auto &[kSinkPart, value1]: cI1S1Iz0Szn1_ ) {
        for ( const auto &[kSinkHole, value2]: value1 ) {
            for ( const auto &[kSrcHole, value3]: value2 ) {
                for ( const auto &[kSrcPart, value4]: value3 ) {

                    for (int bSinkPart=0; bSinkPart<bDim; bSinkPart++) {
                        for (int bSinkHole=0; bSinkHole<bDim; bSinkHole++) {
                            for (int bSrcHole=0; bSrcHole<bDim; bSrcHole++) {
                                for (int bSrcPart=0; bSrcPart<bDim; bSrcPart++) {

                                    cI1S1Iz0Szn1_[kSinkPart][kSrcPart][kSinkHole][kSrcHole](NSL::Slice(), bSinkPart, bSrcPart, bSinkHole, bSrcHole)
                                    
                                    += (0.5 * ((corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart) 
                                    * corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole)
                                    
                                    - corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole) 
                                    * corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart)
                                    
                                    - corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole) 
                                    * corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart)
                                    
                                    + corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart) 
                                    * corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole)) 
                                    
                                    / Type(params_["Number Time Sources"])));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Izn1Sz1_() {
    int kDim = params_["wallSources"].shape(0).to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).to<NSL::size_t>();
    int Nt = params_["Nt"].to<NSL::size_t>();

    std::vector<NSL::size_t> tDim{0};
    
    // NSL::Tensor<Type> eye(params_["device"].to<NSL::Device>(),kDim,kDim,Nt,bDim,bDim);
    // for (int k=0; k<kDim; k++) {
    //     for (int nt=0;nt<Nt; nt++){
    //         eye(k,k,nt,NSL::Slice(), NSL::Slice()) = NSL::Matrix::Identity<Type> (bDim);
    //     }
    // }

    for ( const auto &[kSinkPart, value1]: cI1S1Izn1Sz1_ ) {
        for ( const auto &[kSinkHole, value2]: value1 ) {
            for ( const auto &[kSrcHole, value3]: value2 ) {
                for ( const auto &[kSrcPart, value4]: value3 ) {

                    for (int bSinkPart=0; bSinkPart<bDim; bSinkPart++) {
                        for (int bSinkHole=0; bSinkHole<bDim; bSinkHole++) {
                            for (int bSrcHole=0; bSrcHole<bDim; bSrcHole++) {
                                for (int bSrcPart=0; bSrcPart<bDim; bSrcPart++) {

                                    cI1S1Izn1Sz1_[kSinkPart][kSrcPart][kSinkHole][kSrcHole](NSL::Slice(), bSinkPart, bSrcPart, bSinkHole, bSrcHole)
                                    
                                    += ((NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart), tDim) 
                                    * NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole), tDim)
                                    
                                    - NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole), tDim) 
                                    * NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart), tDim))

                                    // + eye(kSinkHole, kSrcHole, NSL::Slice(), bSinkHole, bSrcHole) * corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart)

                                    / Type(params_["Number Time Sources"]));
                                }
                            }
                        }
                    } 
                }
            }
        }
    }
}

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Izn1Sz0_() {
    int kDim = params_["wallSources"].shape(0).to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).to<NSL::size_t>();
    int Nt = params_["Nt"].to<NSL::size_t>();

    std::vector<NSL::size_t> tDim{0};

    for ( const auto &[kSinkPart, value1]: cI1S1Izn1Sz0_ ) {
        for ( const auto &[kSinkHole, value2]: value1 ) {
            for ( const auto &[kSrcHole, value3]: value2 ) {
                for ( const auto &[kSrcPart, value4]: value3 ) {

                    for (int bSinkPart=0; bSinkPart<bDim; bSinkPart++) {
                        for (int bSinkHole=0; bSinkHole<bDim; bSinkHole++) {
                            for (int bSrcHole=0; bSrcHole<bDim; bSrcHole++) {
                                for (int bSrcPart=0; bSrcPart<bDim; bSrcPart++) {

                                    cI1S1Izn1Sz0_[kSinkPart][kSrcPart][kSinkHole][kSrcHole](NSL::Slice(), bSinkPart, bSrcPart, bSinkHole, bSrcHole)
                                    
                                    += -1*(0.5 * ((corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart) 
                                    * NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole), tDim)
                                    
                                    - corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole) 
                                    * NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart), tDim)
                                    
                                    - NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole), tDim) 
                                    * corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart)
                                    
                                    + NSL::LinAlg::flip(corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart), tDim) 
                                    * corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole)) 
                                    
                                    / Type(params_["Number Time Sources"])));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Izn1Szn1_() {
    int kDim = params_["wallSources"].shape(0).to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).to<NSL::size_t>();
    int Nt = params_["Nt"].to<NSL::size_t>();
    
    // NSL::Tensor<Type> eye(params_["device"].to<NSL::Device>(),kDim,kDim,Nt,bDim,bDim);
    // for (int k=0; k<kDim; k++) {
    //     for (int nt=0;nt<Nt; nt++){
    //         eye(k,k,nt,NSL::Slice(), NSL::Slice()) = NSL::Matrix::Identity<Type> (bDim);
    //     }
    // }

    for ( const auto &[kSinkPart, value1]: cI1S1Izn1Szn1_ ) {
        for ( const auto &[kSinkHole, value2]: value1 ) {
            for ( const auto &[kSrcHole, value3]: value2 ) {
                for ( const auto &[kSrcPart, value4]: value3 ) {

                    for (int bSinkPart=0; bSinkPart<bDim; bSinkPart++) {
                        for (int bSinkHole=0; bSinkHole<bDim; bSinkHole++) {
                            for (int bSrcHole=0; bSrcHole<bDim; bSrcHole++) {
                                for (int bSrcPart=0; bSrcPart<bDim; bSrcPart++) {

                                    cI1S1Izn1Szn1_[kSinkPart][kSrcPart][kSinkHole][kSrcHole](NSL::Slice(), bSinkPart, bSrcPart, bSinkHole, bSrcHole)
                                    
                                    += ((corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart) 
                                    * corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole)
                                    
                                    - corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole) 
                                    * corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart))

                                    // + eye(kSinkHole, kSrcHole, NSL::Slice(), bSinkHole, bSrcHole) * corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart)

                                    / Type(params_["Number Time Sources"]));
                                }
                            }
                        }
                    } 
                }
            }
        }
    }
}

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I0S1Iz0Szn1_() {
    int kDim = params_["wallSources"].shape(0).to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).to<NSL::size_t>();
    int Nt = params_["Nt"].to<NSL::size_t>();

    for ( const auto &[kSinkPart, value1]: cI0S1Iz0Szn1_ ) {
        for ( const auto &[kSinkHole, value2]: value1 ) {
            for ( const auto &[kSrcHole, value3]: value2 ) {
                for ( const auto &[kSrcPart, value4]: value3 ) {

                    for (int bSinkPart=0; bSinkPart<bDim; bSinkPart++) {
                        for (int bSinkHole=0; bSinkHole<bDim; bSinkHole++) {
                            for (int bSrcHole=0; bSrcHole<bDim; bSrcHole++) {
                                for (int bSrcPart=0; bSrcPart<bDim; bSrcPart++) {

                                    cI0S1Iz0Szn1_[kSinkPart][kSrcPart][kSinkHole][kSrcHole](NSL::Slice(), bSinkPart, bSrcPart, bSinkHole, bSrcHole)
                                    
                                    += (0.5 * ((corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart) 
                                    * corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole)
                                    
                                    + corrPool_[NSL::Hubbard::Particle](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole) 
                                    * corrPool_[NSL::Hubbard::Hole](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart)
                                    
                                    + corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcHole,NSL::Slice(),bSinkPart,bSrcHole) 
                                    * corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcPart,NSL::Slice(),bSinkHole,bSrcPart)
                                    
                                    + corrPool_[NSL::Hubbard::Hole](kSinkPart,kSrcPart,NSL::Slice(),bSinkPart,bSrcPart) 
                                    * corrPool_[NSL::Hubbard::Particle](kSinkHole,kSrcHole,NSL::Slice(),bSinkHole,bSrcHole)) 
                                    
                                    / Type(params_["Number Time Sources"])));

                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace NSL::Measure::Hubbard

#endif // NSL_TWO_POINT_CORRELATION_FUNCTION_TPP
