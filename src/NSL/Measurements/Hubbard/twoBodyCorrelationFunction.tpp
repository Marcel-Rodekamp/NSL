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
            cgDag_(hfm_, NSL::FermionMatrix::MdaggerM),
            corrK_(
                params["device"].template to<NSL::Device>(),
                params["wallSources"].shape(1).template to<NSL::size_t>(),
                params["Nt"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>()
                ),
            corrKDag_(
                params["device"].template to<NSL::Device>(),
                params["wallSources"].shape(1).template to<NSL::size_t>(),
                params["Nt"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>()
                ),
            corrKPool_(
                params["device"].template to<NSL::Device>(),
                params["wallSources"].shape(0).template to<NSL::size_t>(),
                params["wallSources"].shape(0).template to<NSL::size_t>(),
                // NSL::size_t (species_.size()), // I want to have a size 2 in this place
                params["Nt"].template to<NSL::size_t>(),
                params["wallSources"].shape(1).template to<NSL::size_t>(),
                params["wallSources"].shape(1).template to<NSL::size_t>()
                ),
            corrKPoolDag_(
                params["device"].template to<NSL::Device>(),
                params["wallSources"].shape(0).template to<NSL::size_t>(),
                params["wallSources"].shape(0).template to<NSL::size_t>(),
                // NSL::size_t (species_.size()), // I want to have a size 2 in this place
                params["Nt"].template to<NSL::size_t>(),
                params["wallSources"].shape(1).template to<NSL::size_t>(),
                params["wallSources"].shape(1).template to<NSL::size_t>()
                ),
	        srcVecK_(
                params["device"].template to<NSL::Device>(),
                params["wallSources"].shape(1).template to<NSL::size_t>(),
                params["Nt"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>()
            ),
            phi_(
                params["device"].template to<NSL::Device>(),
                params["Nt"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>()
            ),
            basenode_(basenode_),
            hfmPre_(lattice, params),
            cgPre_(hfm_, hfmPre_, NSL::FermionMatrix::MMdagger),
            cgDagPre_(hfm_, hfmPre_, NSL::FermionMatrix::MdaggerM),
            phiPre_(
                params["device"].template to<NSL::Device>(),
                params["Nt"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>()
            ),
            saveCfg_(
                params["device"].template to<NSL::Device>(),
                params["Number Time Sources"].template to<NSL::size_t>(),
                params["wallSources"].shape(0).template to<NSL::size_t>(),
                2,
                params["wallSources"].shape(1).template to<NSL::size_t>(),
                params["Nt"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>()
            ),
            saveCfgDag_(
                params["device"].template to<NSL::Device>(),
                params["Number Time Sources"].template to<NSL::size_t>(),
                params["wallSources"].shape(0).template to<NSL::size_t>(),
                2,
                params["wallSources"].shape(1).template to<NSL::size_t>(),
                params["Nt"].template to<NSL::size_t>(),
                params["Nx"].template to<NSL::size_t>()
            )
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

        bool eq_modBZ_(const NSL::Tensor<double>& k, const NSL::Tensor<double>& q, double eps = 2e-12) {
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

            return (std::abs(std::round(m)- m) < eps && std::abs(std::round(n)- n) < eps);

            // return false;
        }

        // NSL::Tensor<Type> kroneckerProduct_(const NSL::Tensor<Type>& A, const NSL::Tensor<Type>& B) {
            
        //     int rowsA = A.shape(1);
        //     int colsA = A.shape(2);
        //     int rowsB = B.shape(1);
        //     int colsB = B.shape(2);

        //     // Result Tensor has shape ( (rowsA*rowsB) x (colsA*colsB) )
        //     NSL::Tensor<Type> result(params_["device"].template to<NSL::Device>(), A.shape(0), rowsA * rowsB, colsA * colsB);

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
        NSL::LinAlg::CG<Type> cgDag_;

        FermionMatrixType hfmPre_;
        NSL::LinAlg::CGpreconditioned<Type> cgPre_;
        NSL::LinAlg::CGpreconditioned<Type> cgDagPre_;
        NSL::Tensor<Type> phiPre_;
        NSL::Tensor<Type> saveCfg_;
        NSL::Tensor<Type> saveCfgDag_;

        NSL::Tensor<Type> corrK_;
        NSL::Tensor<Type> corrKDag_;
        NSL::Tensor<Type> corrKPool_;
        std::unordered_map<NSL::Hubbard::Species, NSL::Tensor<Type>> corrPool_;
        NSL::Tensor<Type> corrKPoolDag_;
        std::unordered_map<NSL::Hubbard::Species, NSL::Tensor<Type>> corrPoolDag_;
        NSL::Tensor<Type> srcVecK_;

        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Iz1Sz1_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Iz1Sz0_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Iz1Szn1_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Iz0Sz1_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Iz0Szn1_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Izn1Sz1_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Izn1Sz0_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S1Izn1Szn1_;

        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI0S1Iz0Sz1_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI0S1Iz0Szn1_;

        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S0Iz1Sz0_;
        std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, std::unordered_map<NSL::size_t, NSL::Tensor<Type>>>>> cI1S0Izn1Sz0_;

        NSL::Tensor<Type> phi_;

        std::string basenode_;

        void I1S1Iz1Sz1_(NSL::size_t NumberTimeSources);
        void I1S1Iz1Sz0_(NSL::size_t NumberTimeSources);
        void I1S1Iz1Szn1_(NSL::size_t NumberTimeSources);
        void I1S1Iz0Sz1_(NSL::size_t NumberTimeSources);
        void I1S1Iz0Szn1_(NSL::size_t NumberTimeSources);
        void I1S1Izn1Sz1_(NSL::size_t NumberTimeSources);
        void I1S1Izn1Sz0_(NSL::size_t NumberTimeSources);
        void I1S1Izn1Szn1_(NSL::size_t NumberTimeSources);

        void I0S1Iz0Sz1_(NSL::size_t NumberTimeSources);
        void I0S1Iz0Szn1_(NSL::size_t NumberTimeSources);

        void I1S0Iz1Sz0_(NSL::size_t NumberTimeSources);
        void I1S0Izn1Sz0_(NSL::size_t NumberTimeSources);
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
    for (int w=0; w<kDim; w++) {
        for (int x=0; x<kDim; x++) {
            for (int y=0; y<kDim; y++) {
                for (int z=0; z<kDim; z++) {
                    if (!(eq_modBZ_(momenta(w,NSL::Slice()) + momenta(x,NSL::Slice()), momenta(z, NSL::Slice()) + momenta(y,NSL::Slice())))) {
                        continue;
                    }
                    cI1S1Iz1Sz1_[w][x][z][y] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), Nt, bDim * bDim, bDim * bDim);
                    cI1S1Iz1Sz0_[w][x][z][y] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), Nt, bDim * bDim, bDim * bDim);
                    cI1S1Iz1Szn1_[w][x][z][y] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), Nt, bDim * bDim, bDim * bDim);
                    cI1S1Iz0Sz1_[w][x][z][y] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), Nt, bDim * bDim, bDim * bDim);
                    cI1S1Iz0Szn1_[w][x][z][y] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), Nt, bDim * bDim, bDim * bDim);
                    cI1S1Izn1Sz1_[w][x][z][y] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), Nt, bDim * bDim, bDim * bDim);
                    cI1S1Izn1Sz0_[w][x][z][y] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), Nt, bDim * bDim, bDim * bDim);
                    cI1S1Izn1Szn1_[w][x][z][y] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), Nt, bDim * bDim, bDim * bDim);

                    cI0S1Iz0Sz1_[w][x][z][y] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), Nt, bDim * bDim, bDim * bDim);
                    cI0S1Iz0Szn1_[w][x][z][y] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), Nt, bDim * bDim, bDim * bDim);

                    cI1S0Iz1Sz0_[w][x][z][y] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), Nt, bDim * bDim, bDim * bDim);
                    cI1S0Izn1Sz0_[w][x][z][y] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), Nt, bDim * bDim, bDim * bDim);
                }
            }
        }
    }
    corrPool_[NSL::Hubbard::Particle] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), kDim, kDim, Nt, bDim, bDim);
    corrPool_[NSL::Hubbard::Hole] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), kDim, kDim, Nt, bDim, bDim);
    corrPoolDag_[NSL::Hubbard::Particle] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), kDim, kDim, Nt, bDim, bDim);
    corrPoolDag_[NSL::Hubbard::Hole] = NSL::Tensor<Type> (params_["device"].template to<NSL::Device>(), kDim, kDim, Nt, bDim, bDim);

    NSL::size_t tID = 0;
    for (NSL::size_t tsrc = 0; tsrc<Nt; tsrc+=tsrcStep) {
        NSL::size_t i = 0;
        for (NSL::Hubbard::Species species : {NSL::Hubbard::Particle, NSL::Hubbard::Hole}) {
            // Reset memory
            // - Result correlator
            corrKPool_ = Type(0);

            // populate the fermion matrix using the free configuration
            hfmPre_.populate(phiPre_,species);

            // populate the fermion matrix using the free configuration
            hfm_.populate(phi_,species);

            for (int kSrc=0; kSrc<kDim; kSrc++ ) {

                // Define a wall source
                srcVecK_(NSL::Slice(),tsrc,NSL::Slice()) = NSL::Tensor<NSL::complex<double>> (params_["wallSources"])(kSrc,NSL::Slice(),NSL::Slice());
                // invert MM^dagger pre
                NSL::Tensor<Type> invMMdag = cgPre_(srcVecK_, saveCfg_(tID, kSrc, i, NSL::Ellipsis()));
                // invert M^daggerM pre
                NSL::Tensor<Type> invMdagM = cgDagPre_(srcVecK_, saveCfgDag_(tID, kSrc, i, NSL::Ellipsis()));

                // // invert MM^dagger pre
                // NSL::Tensor<Type> invMMdagPre = cg_(srcVecK_, srcVecK_);
                // // invert M^daggerM pre
                // NSL::Tensor<Type> invMdagMPre = cgDag_(srcVecK_, srcVecK_);

                // invMMdagPre += 1.000001;
                // invMdagMPre += 1.000001;
                
                // // invert MM^dagger pre
                // NSL::Tensor<Type> invMMdag = cg_(srcVecK_, invMMdagPre);
                // // invert M^daggerM pre
                // NSL::Tensor<Type> invMdagM = cgDag_(srcVecK_, invMdagMPre);

                saveCfg_(tID, kSrc, i, NSL::Ellipsis()) = Type(0);
                saveCfgDag_(tID, kSrc, i, NSL::Ellipsis()) = Type(0);

                saveCfg_(tID, kSrc, i, NSL::Ellipsis()) = invMMdag;
                saveCfgDag_(tID, kSrc, i, NSL::Ellipsis()) = invMdagM;

                // saveCfg_(
                // params["device"].template to<NSL::Device>(),
                // params["Number Time Sources"].template to<NSL::size_t>(),
                // params["wallSources"].shape(0).template to<NSL::size_t>(),
                // 2,
                // params["wallSources"].shape(1).template to<NSL::size_t>(),
                // params["Nt"].template to<NSL::size_t>(),
                // params["Nx"].template to<NSL::size_t>()
            // )
                // // invert MM^dagger pre
                // // std::cout << "HERE" << std::endl;
                // NSL::Tensor<Type> invMMdagPre = cgPre_(srcVecK_ - hfm_.MMdagger(srcVecK_));
                // // invert M^daggerM pre
                // NSL::Tensor<Type> invMdagMPre = cgDagPre_(srcVecK_ - hfm_.MdaggerM(srcVecK_));

                // // for (int preStep=0; preStep<100; preStep++ ){
                // //     invMMdagPre = hfm_.MMdagger(invMMdagPre);
                // //     invMdagMPre = hfm_.MdaggerM(invMdagMPre);
                // // }

                // // invert MM^dagger
                // NSL::Tensor<Type> invMMdag = cg_(srcVecK_, invMMdagPre);
                // // invert M^daggerM
                // NSL::Tensor<Type> invMdagM = cgDag_(srcVecK_, invMdagMPre);

                // Reset memory
                // - Source vector
                srcVecK_ = Type(0);

                // back multiply M^dagger to obtain M^{-1}
                // invM is of shape Nx x Nt x Nx
                NSL::Tensor<Type> invM = hfm_.Mdagger(invMMdag);
                NSL::Tensor<Type> invMdagger = hfm_.M(invMdagM);

                // We shift the 1st axis (time-axis) if invM by tsrc and apply anti periodic 
                // boundary conditions
                // shift t -> t - tsrc
                invM.shift( -tsrc, -2, -Type(1) );
                invMdagger.shift( -tsrc, -2, -Type(1) );

                // Average over all source times
                // corrK_ += invM; // I changed something here!!!!!
                corrK_ = invM;
                corrKDag_ = invMdagger;

                for (int kSink=0; kSink<kDim; kSink++) {
                    for (NSL::size_t t = 0; t<Nt; t++) {
                        for (int sigmaSink=0; sigmaSink<bDim; sigmaSink++) {
                            for (int sigmaSrc=0; sigmaSrc<bDim; sigmaSrc++) {
                                corrKPool_(kSink,kSrc,t,sigmaSink,sigmaSrc) = NSL::LinAlg::inner_product( NSL::Tensor<NSL::complex<double>> (params_["wallSources"])(kSink,sigmaSink,NSL::Slice()), corrK_(sigmaSrc,t,NSL::Slice()));

                                corrKPoolDag_(kSink,kSrc,t,sigmaSink,sigmaSrc) = NSL::LinAlg::inner_product( NSL::Tensor<NSL::complex<double>> (params_["wallSources"])(kSink,sigmaSink,NSL::Slice()), corrKDag_(sigmaSrc,t,NSL::Slice()));
                            }
                        }
                    }
                }
                // Reset memory
                // - Result correlator
                corrK_ = Type(0);
                corrKDag_ = Type(0);
            } // for kSrc

            corrPool_[species] = corrKPool_;
            corrPoolDag_[species] = corrKPoolDag_.conj();
            i++;
        } // for species

        I1S1Iz1Sz1_(NumberTimeSources);
        I1S1Iz1Sz0_(NumberTimeSources);
        I1S1Iz1Szn1_(NumberTimeSources);
        I1S1Iz0Sz1_(NumberTimeSources);
        I1S1Iz0Szn1_(NumberTimeSources);
        I1S1Izn1Sz1_(NumberTimeSources);
        I1S1Izn1Sz0_(NumberTimeSources);
        I1S1Izn1Szn1_(NumberTimeSources);

        I0S1Iz0Sz1_(NumberTimeSources);
        I0S1Iz0Szn1_(NumberTimeSources);

        I1S0Iz1Sz0_(NumberTimeSources);
        I1S0Izn1Sz0_(NumberTimeSources);

        tID++;
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

    phiPre_ = Type(0);

    // write the momenta out
    if (!h5_.exist(std::string(basenode_)+"/Momenta")){
        NSL::Tensor<double> momenta = params_["momenta"];
        this->h5_.write(momenta,std::string(basenode_)+"/Momenta");
    }

    // write the non interacting correlator 
    std::string node;
    node = "/NonInteracting/correlators/twobody";
    // this is a shortcut, we don't need to calculate the non-interacting 
    // correlators if we won't update the file
    if(!skip_(this->params_["overwrite"],node)) {
        // measure the non-interacting theory
    	// U = 0 <=> phi = 0
    	phi_ = Type(0);

    	measure(1);
    	std::string momNode;
    	for ( const auto &[w, value1]: cI1S1Iz1Sz1_ ) {
            for ( const auto &[x, value2]: value1 ) {
            	for ( const auto &[y, value3]: value2 ) {
                    for ( const auto &[z, value4]: value3 ) {
                    	// write the result
                    	momNode = fmt::format("/cI1S1Iz1Sz1/{}-{}-{}-{}",w,x,z,y);
                    	this->h5_.write(cI1S1Iz1Sz1_[w][x][z][y],std::string(basenode_)+node+momNode);

                   	 momNode = fmt::format("/cI1S1Iz1Sz0/{}-{}-{}-{}",w,x,z,y);
                    	 this->h5_.write(cI1S1Iz1Sz0_[w][x][z][y],std::string(basenode_)+node+momNode);

                    	 momNode = fmt::format("/cI1S1Iz1Szn1/{}-{}-{}-{}",w,x,z,y);
                    	 this->h5_.write(cI1S1Iz1Szn1_[w][x][z][y],std::string(basenode_)+node+momNode);

                    	 momNode = fmt::format("/cI1S1Iz0Sz1/{}-{}-{}-{}",w,x,z,y);
                    	 this->h5_.write(cI1S1Iz0Sz1_[w][x][z][y],std::string(basenode_)+node+momNode);

                    	 momNode = fmt::format("/cI1S1Iz0Szn1/{}-{}-{}-{}",w,x,z,y);
                    	 this->h5_.write(cI1S1Iz0Szn1_[w][x][z][y],std::string(basenode_)+node+momNode);

                    	 momNode = fmt::format("/cI1S1Izn1Sz1/{}-{}-{}-{}",w,x,z,y);
                    	 this->h5_.write(cI1S1Izn1Sz1_[w][x][z][y],std::string(basenode_)+node+momNode);

                    	 momNode = fmt::format("/cI1S1Izn1Sz0/{}-{}-{}-{}",w,x,z,y);
                    	 this->h5_.write(cI1S1Izn1Sz0_[w][x][z][y],std::string(basenode_)+node+momNode);

                    	 momNode = fmt::format("/cI1S1Izn1Szn1/{}-{}-{}-{}",w,x,z,y);
                    	 this->h5_.write(cI1S1Izn1Szn1_[w][x][z][y],std::string(basenode_)+node+momNode);

                    	 momNode = fmt::format("/cI0S1Iz0Sz1/{}-{}-{}-{}",w,x,z,y);
                    	 this->h5_.write(cI0S1Iz0Sz1_[w][x][z][y],std::string(basenode_)+node+momNode);

			 momNode = fmt::format("/cI0S1Iz0Szn1/{}-{}-{}-{}",w,x,z,y);
                    	 this->h5_.write(cI0S1Iz0Szn1_[w][x][z][y],std::string(basenode_)+node+momNode);

                    	 momNode = fmt::format("/cI1S0Iz1Sz0/{}-{}-{}-{}",w,x,z,y);
                    	 this->h5_.write(cI1S0Iz1Sz0_[w][x][z][y],std::string(basenode_)+node+momNode);

			 momNode = fmt::format("/cI1S0Izn1Sz0/{}-{}-{}-{}",w,x,z,y);
                    	 this->h5_.write(cI1S0Izn1Sz0_[w][x][z][y],std::string(basenode_)+node+momNode);
                	 }
            	     }
       		 }
    	    }
        } else {
        NSL::Logger::info("Non-interacting correlators already exist");
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
        for ( const auto &[w, value1]: cI1S1Iz1Sz1_ ) {
            for ( const auto &[x, value2]: value1 ) {
                for ( const auto &[y, value3]: value2 ) {
                    for ( const auto &[z, value4]: value3 ) {
                        // write the result
                        momNode = fmt::format("/cI1S1Iz1Sz1/{}-{}-{}-{}",w,x,z,y);
                        this->h5_.write(cI1S1Iz1Sz1_[w][x][z][y],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Iz1Sz0/{}-{}-{}-{}",w,x,z,y);
                        this->h5_.write(cI1S1Iz1Sz0_[w][x][z][y],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Iz1Szn1/{}-{}-{}-{}",w,x,z,y);
                        this->h5_.write(cI1S1Iz1Szn1_[w][x][z][y],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Iz0Sz1/{}-{}-{}-{}",w,x,z,y);
                        this->h5_.write(cI1S1Iz0Sz1_[w][x][z][y],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Iz0Szn1/{}-{}-{}-{}",w,x,z,y);
                        this->h5_.write(cI1S1Iz0Szn1_[w][x][z][y],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Izn1Sz1/{}-{}-{}-{}",w,x,z,y);
                        this->h5_.write(cI1S1Izn1Sz1_[w][x][z][y],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Izn1Sz0/{}-{}-{}-{}",w,x,z,y);
                        this->h5_.write(cI1S1Izn1Sz0_[w][x][z][y],std::string(basenode_)+node+momNode);

                        momNode = fmt::format("/cI1S1Izn1Szn1/{}-{}-{}-{}",w,x,z,y);
                        this->h5_.write(cI1S1Izn1Szn1_[w][x][z][y],std::string(basenode_)+node+momNode);


                        momNode = fmt::format("/cI0S1Iz0Sz1/{}-{}-{}-{}",w,x,z,y);
                        this->h5_.write(cI0S1Iz0Sz1_[w][x][z][y],std::string(basenode_)+node+momNode);
                        momNode = fmt::format("/cI0S1Iz0Szn1/{}-{}-{}-{}",w,x,z,y);
                        this->h5_.write(cI0S1Iz0Szn1_[w][x][z][y],std::string(basenode_)+node+momNode);


                        momNode = fmt::format("/cI1S0Iz1Sz0/{}-{}-{}-{}",w,x,z,y);
                        this->h5_.write(cI1S0Iz1Sz0_[w][x][z][y],std::string(basenode_)+node+momNode);
                        momNode = fmt::format("/cI1S0Izn1Sz0/{}-{}-{}-{}",w,x,z,y);
                        this->h5_.write(cI1S0Izn1Sz0_[w][x][z][y],std::string(basenode_)+node+momNode);
                    }
                }
            }
        }
        
    } // for cfgI

} // measure()

// The labels of the next functions might look incorrect but they infact are the correct one. 
// We need to read the last two labels in the opposite direction because they are part of the opperator (O_{kl})^\dagger where the dagger swaps their place.
// For further information check Petar's Excitons Notes

template<
    NSL::Concept::isNumber Type,
    NSL::Concept::isDerived<NSL::Lattice::SpatialLattice<Type>> LatticeType,
    NSL::Concept::isDerived<NSL::FermionMatrix::FermionMatrix<Type,LatticeType>> FermionMatrixType
>
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Iz1Sz1_(NSL::size_t NumberTimeSources) {
    int kDim = params_["wallSources"].shape(0).template to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).template to<NSL::size_t>();
    int Nt = params_["Nt"].template to<NSL::size_t>();

    // NSL::Tensor<Type> eye(params_["device"].template to<NSL::Device>(),kDim,kDim,Nt,bDim,bDim);
    // for (int k=0; k<kDim; k++) {
    //     for (int nt=0;nt<Nt; nt++){
    //         eye(k,k,nt,NSL::Slice(), NSL::Slice()) = NSL::Matrix::Identity<Type> (bDim);
    //     }
    // }

    for ( const auto &[w, value1]: cI1S1Iz1Sz1_ ) {
        for ( const auto &[x, value2]: value1 ) {
            for ( const auto &[y, value3]: value2 ) {
                for ( const auto &[z, value4]: value3 ) {

                    for (int i=0; i<bDim; i++) {
                        for (int j=0; j<bDim; j++) {
                            for (int k=0; k<bDim; k++) {
                                for (int l=0; l<bDim; l++) {

                                    cI1S1Iz1Sz1_[w][x][z][y](NSL::Slice(), i * bDim + j, l * bDim + k)
                                    
                                    += ((corrPoolDag_[NSL::Hubbard::Particle](w,z,NSL::Slice(),i,l) 
                                    * corrPoolDag_[NSL::Hubbard::Particle](x,y,NSL::Slice(),j,k)
                                    
                                    - corrPoolDag_[NSL::Hubbard::Particle](w,y,NSL::Slice(),i,k)
                                    * corrPoolDag_[NSL::Hubbard::Particle](x,z,NSL::Slice(),j,l))

                                    // + eye(x, y, NSL::Slice(), j, k) * corrPool_[NSL::Hubbard::Particle](w,z,NSL::Slice(),i,l)

                                    / Type(NumberTimeSources));
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
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Iz1Sz0_(NSL::size_t NumberTimeSources) {
    int kDim = params_["wallSources"].shape(0).template to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).template to<NSL::size_t>();
    int Nt = params_["Nt"].template to<NSL::size_t>();

    for ( const auto &[w, value1]: cI1S1Iz1Sz0_ ) {
        for ( const auto &[x, value2]: value1 ) {
            for ( const auto &[y, value3]: value2 ) {
                for ( const auto &[z, value4]: value3 ) {

                    for (int i=0; i<bDim; i++) {
                        for (int j=0; j<bDim; j++) {
                            for (int k=0; k<bDim; k++) {
                                for (int l=0; l<bDim; l++) {

                                    cI1S1Iz1Sz0_[w][x][z][y](NSL::Slice(), i * bDim + j, l * bDim + k)
                                    
                                    += (-0.5 * ((corrPoolDag_[NSL::Hubbard::Particle](w,z,NSL::Slice(),i,l)
                                    * corrPool_[NSL::Hubbard::Hole](x,y,NSL::Slice(),j,k)
                                    
                                    - corrPoolDag_[NSL::Hubbard::Particle](w,y,NSL::Slice(),i,k)
                                    * corrPool_[NSL::Hubbard::Hole](x,z,NSL::Slice(),j,l)
                                    
                                    - corrPool_[NSL::Hubbard::Hole](w,y,NSL::Slice(),i,k) 
                                    * corrPoolDag_[NSL::Hubbard::Particle](x,z,NSL::Slice(),j,l)
                                    
                                    + corrPool_[NSL::Hubbard::Hole](w,z,NSL::Slice(),i,l) 
                                    * corrPoolDag_[NSL::Hubbard::Particle](x,y,NSL::Slice(),j,k)) 
                                    
                                    / Type(NumberTimeSources)));
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
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Iz1Szn1_(NSL::size_t NumberTimeSources) {
    int kDim = params_["wallSources"].shape(0).template to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).template to<NSL::size_t>();
    int Nt = params_["Nt"].template to<NSL::size_t>();

    for ( const auto &[w, value1]: cI1S1Iz1Szn1_ ) {
        for ( const auto &[x, value2]: value1 ) {
            for ( const auto &[y, value3]: value2 ) {
                for ( const auto &[z, value4]: value3 ) {

                    for (int i=0; i<bDim; i++) {
                        for (int j=0; j<bDim; j++) {
                            for (int k=0; k<bDim; k++) {
                                for (int l=0; l<bDim; l++) {

                                    cI1S1Iz1Szn1_[w][x][z][y](NSL::Slice(), i * bDim + j, l * bDim + k)
                                    
                                    += ((corrPool_[NSL::Hubbard::Hole](w,z,NSL::Slice(),i,l) 
                                    * corrPool_[NSL::Hubbard::Hole](x,y,NSL::Slice(),j,k)
                                    
                                    - corrPool_[NSL::Hubbard::Hole](w,y,NSL::Slice(),i,k) 
                                    * corrPool_[NSL::Hubbard::Hole](x,z,NSL::Slice(),j,l))

                                    / Type(NumberTimeSources));
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
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Iz0Sz1_(NSL::size_t NumberTimeSources) {
    int kDim = params_["wallSources"].shape(0).template to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).template to<NSL::size_t>();
    int Nt = params_["Nt"].template to<NSL::size_t>();

    for ( const auto &[w, value1]: cI1S1Iz0Sz1_ ) {
        for ( const auto &[x, value2]: value1 ) {
            for ( const auto &[y, value3]: value2 ) {
                for ( const auto &[z, value4]: value3 ) {

                    for (int i=0; i<bDim; i++) {
                        for (int j=0; j<bDim; j++) {
                            for (int k=0; k<bDim; k++) {
                                for (int l=0; l<bDim; l++) {

                                    cI1S1Iz0Sz1_[w][x][z][y](NSL::Slice(), i * bDim + j, l * bDim + k)
                                    
                                    += (0.5 * ((corrPoolDag_[NSL::Hubbard::Particle](w,z,NSL::Slice(),i,l)
                                    * corrPoolDag_[NSL::Hubbard::Hole](x,y,NSL::Slice(),j,k)
                                    
                                    - corrPoolDag_[NSL::Hubbard::Particle](w,y,NSL::Slice(),i,k)
                                    * corrPoolDag_[NSL::Hubbard::Hole](x,z,NSL::Slice(),j,l)
                                    
                                    - corrPoolDag_[NSL::Hubbard::Hole](w,y,NSL::Slice(),i,k)
                                    * corrPoolDag_[NSL::Hubbard::Particle](x,z,NSL::Slice(),j,l)
                                    
                                    + corrPoolDag_[NSL::Hubbard::Hole](w,z,NSL::Slice(),i,l)
                                    * corrPoolDag_[NSL::Hubbard::Particle](x,y,NSL::Slice(),j,k)) 
                                    
                                    / Type(NumberTimeSources)));
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
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Iz0Szn1_(NSL::size_t NumberTimeSources) {
    int kDim = params_["wallSources"].shape(0).template to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).template to<NSL::size_t>();
    int Nt = params_["Nt"].template to<NSL::size_t>();

    for ( const auto &[w, value1]: cI1S1Iz0Szn1_ ) {
        for ( const auto &[x, value2]: value1 ) {
            for ( const auto &[y, value3]: value2 ) {
                for ( const auto &[z, value4]: value3 ) {

                    for (int i=0; i<bDim; i++) {
                        for (int j=0; j<bDim; j++) {
                            for (int k=0; k<bDim; k++) {
                                for (int l=0; l<bDim; l++) {

                                    cI1S1Iz0Szn1_[w][x][z][y](NSL::Slice(), i * bDim + j, l * bDim + k)
                                    
                                    += (0.5 * ((corrPool_[NSL::Hubbard::Particle](w,z,NSL::Slice(),i,l) 
                                    * corrPool_[NSL::Hubbard::Hole](x,y,NSL::Slice(),j,k)
                                    
                                    - corrPool_[NSL::Hubbard::Particle](w,y,NSL::Slice(),i,k) 
                                    * corrPool_[NSL::Hubbard::Hole](x,z,NSL::Slice(),j,l)
                                    
                                    - corrPool_[NSL::Hubbard::Hole](w,y,NSL::Slice(),i,k) 
                                    * corrPool_[NSL::Hubbard::Particle](x,z,NSL::Slice(),j,l)
                                    
                                    + corrPool_[NSL::Hubbard::Hole](w,z,NSL::Slice(),i,l) 
                                    * corrPool_[NSL::Hubbard::Particle](x,y,NSL::Slice(),j,k)) 
                                    
                                    / Type(NumberTimeSources)));
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
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Izn1Sz1_(NSL::size_t NumberTimeSources) {
    int kDim = params_["wallSources"].shape(0).template to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).template to<NSL::size_t>();
    int Nt = params_["Nt"].template to<NSL::size_t>();

    for ( const auto &[w, value1]: cI1S1Izn1Sz1_ ) {
        for ( const auto &[x, value2]: value1 ) {
            for ( const auto &[y, value3]: value2 ) {
                for ( const auto &[z, value4]: value3 ) {

                    for (int i=0; i<bDim; i++) {
                        for (int j=0; j<bDim; j++) {
                            for (int k=0; k<bDim; k++) {
                                for (int l=0; l<bDim; l++) {

                                    cI1S1Izn1Sz1_[w][x][z][y](NSL::Slice(), i * bDim + j, l * bDim + k)
                                    
                                    += ((corrPoolDag_[NSL::Hubbard::Hole](w,z,NSL::Slice(),i,l) 
                                    * corrPoolDag_[NSL::Hubbard::Hole](x,y,NSL::Slice(),j,k)
                                    
                                    - corrPoolDag_[NSL::Hubbard::Hole](w,y,NSL::Slice(),i,k) 
                                    * corrPoolDag_[NSL::Hubbard::Hole](x,z,NSL::Slice(),j,l))

                                    / Type(NumberTimeSources));
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
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Izn1Sz0_(NSL::size_t NumberTimeSources) {
    int kDim = params_["wallSources"].shape(0).template to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).template to<NSL::size_t>();
    int Nt = params_["Nt"].template to<NSL::size_t>();

    for ( const auto &[w, value1]: cI1S1Izn1Sz0_ ) {
        for ( const auto &[x, value2]: value1 ) {
            for ( const auto &[y, value3]: value2 ) {
                for ( const auto &[z, value4]: value3 ) {

                    for (int i=0; i<bDim; i++) {
                        for (int j=0; j<bDim; j++) {
                            for (int k=0; k<bDim; k++) {
                                for (int l=0; l<bDim; l++) {

                                    cI1S1Izn1Sz0_[w][x][z][y](NSL::Slice(), i * bDim + j, l * bDim + k)
                                    
                                    += (-0.5 * ((corrPool_[NSL::Hubbard::Particle](w,z,NSL::Slice(),i,l) 
                                    * corrPoolDag_[NSL::Hubbard::Hole](x,y,NSL::Slice(),j,k)
                                    
                                    - corrPool_[NSL::Hubbard::Particle](w,y,NSL::Slice(),i,k) 
                                    * corrPoolDag_[NSL::Hubbard::Hole](x,z,NSL::Slice(),j,l)
                                    
                                    - corrPoolDag_[NSL::Hubbard::Hole](w,y,NSL::Slice(),i,k) 
                                    * corrPool_[NSL::Hubbard::Particle](x,z,NSL::Slice(),j,l)
                                    
                                    + corrPoolDag_[NSL::Hubbard::Hole](w,z,NSL::Slice(),i,l) 
                                    * corrPool_[NSL::Hubbard::Particle](x,y,NSL::Slice(),j,k)) 
                                    
                                    / Type(NumberTimeSources)));
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
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S1Izn1Szn1_(NSL::size_t NumberTimeSources) {
    int kDim = params_["wallSources"].shape(0).template to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).template to<NSL::size_t>();
    int Nt = params_["Nt"].template to<NSL::size_t>();

    for ( const auto &[w, value1]: cI1S1Izn1Szn1_ ) {
        for ( const auto &[x, value2]: value1 ) {
            for ( const auto &[y, value3]: value2 ) {
                for ( const auto &[z, value4]: value3 ) {

                    for (int i=0; i<bDim; i++) {
                        for (int j=0; j<bDim; j++) {
                            for (int k=0; k<bDim; k++) {
                                for (int l=0; l<bDim; l++) {

                                    cI1S1Izn1Szn1_[w][x][z][y](NSL::Slice(), i * bDim + j, l * bDim + k)
                                    
                                    += ((corrPool_[NSL::Hubbard::Particle](w,z,NSL::Slice(),i,l) 
                                    * corrPool_[NSL::Hubbard::Particle](x,y,NSL::Slice(),j,k)
                                    
                                    - corrPool_[NSL::Hubbard::Particle](w,y,NSL::Slice(),i,k) 
                                    * corrPool_[NSL::Hubbard::Particle](x,z,NSL::Slice(),j,l))

                                    / Type(NumberTimeSources));
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
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I0S1Iz0Sz1_(NSL::size_t NumberTimeSources) {
    int kDim = params_["wallSources"].shape(0).template to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).template to<NSL::size_t>();
    int Nt = params_["Nt"].template to<NSL::size_t>();

    for ( const auto &[w, value1]: cI0S1Iz0Sz1_ ) {
        for ( const auto &[x, value2]: value1 ) {
            for ( const auto &[y, value3]: value2 ) {
                for ( const auto &[z, value4]: value3 ) {

                    for (int i=0; i<bDim; i++) {
                        for (int j=0; j<bDim; j++) {
                            for (int k=0; k<bDim; k++) {
                                for (int l=0; l<bDim; l++) {

                                    cI0S1Iz0Sz1_[w][x][z][y](NSL::Slice(), i * bDim + j, l * bDim + k)
                                    
                                    += (0.5 * ((corrPoolDag_[NSL::Hubbard::Particle](w,z,NSL::Slice(),i,l) 
                                    * corrPoolDag_[NSL::Hubbard::Hole](x,y,NSL::Slice(),j,k)
                                    
                                    + corrPoolDag_[NSL::Hubbard::Particle](w,y,NSL::Slice(),i,k) 
                                    * corrPoolDag_[NSL::Hubbard::Hole](x,z,NSL::Slice(),j,l)
                                    
                                    + corrPoolDag_[NSL::Hubbard::Hole](w,y,NSL::Slice(),i,k) 
                                    * corrPoolDag_[NSL::Hubbard::Particle](x,z,NSL::Slice(),j,l)
                                    
                                    + corrPoolDag_[NSL::Hubbard::Hole](w,z,NSL::Slice(),i,l) 
                                    * corrPoolDag_[NSL::Hubbard::Particle](x,y,NSL::Slice(),j,k)) 
                                    
                                    / Type(NumberTimeSources)));

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
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I0S1Iz0Szn1_(NSL::size_t NumberTimeSources) {
    int kDim = params_["wallSources"].shape(0).template to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).template to<NSL::size_t>();
    int Nt = params_["Nt"].template to<NSL::size_t>();

    for ( const auto &[w, value1]: cI0S1Iz0Szn1_ ) {
        for ( const auto &[x, value2]: value1 ) {
            for ( const auto &[y, value3]: value2 ) {
                for ( const auto &[z, value4]: value3 ) {

                    for (int i=0; i<bDim; i++) {
                        for (int j=0; j<bDim; j++) {
                            for (int k=0; k<bDim; k++) {
                                for (int l=0; l<bDim; l++) {

                                    cI0S1Iz0Szn1_[w][x][z][y](NSL::Slice(), i * bDim + j, l * bDim + k)
                                    
                                    += (0.5 * ((corrPool_[NSL::Hubbard::Particle](w,z,NSL::Slice(),i,l) 
                                    * corrPool_[NSL::Hubbard::Hole](x,y,NSL::Slice(),j,k)
                                    
                                    + corrPool_[NSL::Hubbard::Particle](w,y,NSL::Slice(),i,k) 
                                    * corrPool_[NSL::Hubbard::Hole](x,z,NSL::Slice(),j,l)
                                    
                                    + corrPool_[NSL::Hubbard::Hole](w,y,NSL::Slice(),i,k) 
                                    * corrPool_[NSL::Hubbard::Particle](x,z,NSL::Slice(),j,l)
                                    
                                    + corrPool_[NSL::Hubbard::Hole](w,z,NSL::Slice(),i,l) 
                                    * corrPool_[NSL::Hubbard::Particle](x,y,NSL::Slice(),j,k)) 
                                    
                                    / Type(NumberTimeSources)));

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
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S0Iz1Sz0_(NSL::size_t NumberTimeSources) {
    int kDim = params_["wallSources"].shape(0).template to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).template to<NSL::size_t>();
    int Nt = params_["Nt"].template to<NSL::size_t>();

    for ( const auto &[w, value1]: cI1S0Iz1Sz0_ ) {
        for ( const auto &[x, value2]: value1 ) {
            for ( const auto &[y, value3]: value2 ) {
                for ( const auto &[z, value4]: value3 ) {

                    for (int i=0; i<bDim; i++) {
                        for (int j=0; j<bDim; j++) {
                            for (int k=0; k<bDim; k++) {
                                for (int l=0; l<bDim; l++) {

                                    cI1S0Iz1Sz0_[w][x][z][y](NSL::Slice(), i * bDim + j, l * bDim + k)
                                    
                                    += (-0.5 * ((corrPoolDag_[NSL::Hubbard::Particle](w,z,NSL::Slice(),i,l) 
                                    * corrPool_[NSL::Hubbard::Hole](x,y,NSL::Slice(),j,k)
                                    
                                    + corrPoolDag_[NSL::Hubbard::Particle](w,y,NSL::Slice(),i,k) 
                                    * corrPool_[NSL::Hubbard::Hole](x,z,NSL::Slice(),j,l)
                                    
                                    + corrPool_[NSL::Hubbard::Hole](w,y,NSL::Slice(),i,k) 
                                    * corrPoolDag_[NSL::Hubbard::Particle](x,z,NSL::Slice(),j,l)
                                    
                                    + corrPool_[NSL::Hubbard::Hole](w,z,NSL::Slice(),i,l) 
                                    * corrPoolDag_[NSL::Hubbard::Particle](x,y,NSL::Slice(),j,k)) 
                                    
                                    / Type(NumberTimeSources)));
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
void TwoBodyCorrelator<Type,LatticeType,FermionMatrixType>::I1S0Izn1Sz0_(NSL::size_t NumberTimeSources) {
    int kDim = params_["wallSources"].shape(0).template to<NSL::size_t>();
    int bDim = params_["wallSources"].shape(1).template to<NSL::size_t>();
    int Nt = params_["Nt"].template to<NSL::size_t>();

    for ( const auto &[w, value1]: cI1S0Izn1Sz0_ ) {
        for ( const auto &[x, value2]: value1 ) {
            for ( const auto &[y, value3]: value2 ) {
                for ( const auto &[z, value4]: value3 ) {

                    for (int i=0; i<bDim; i++) {
                        for (int j=0; j<bDim; j++) {
                            for (int k=0; k<bDim; k++) {
                                for (int l=0; l<bDim; l++) {

                                    cI1S0Izn1Sz0_[w][x][z][y](NSL::Slice(), i * bDim + j, l * bDim + k)
                                    
                                    += (-0.5 * ((corrPool_[NSL::Hubbard::Particle](w,z,NSL::Slice(),i,l) 
                                    * corrPoolDag_[NSL::Hubbard::Hole](x,y,NSL::Slice(),j,k)
                                    
                                    + corrPool_[NSL::Hubbard::Particle](w,y,NSL::Slice(),i,k) 
                                    * corrPoolDag_[NSL::Hubbard::Hole](x,z,NSL::Slice(),j,l)
                                    
                                    + corrPoolDag_[NSL::Hubbard::Hole](w,y,NSL::Slice(),i,k) 
                                    * corrPool_[NSL::Hubbard::Particle](x,z,NSL::Slice(),j,l)
                                    
                                    + corrPoolDag_[NSL::Hubbard::Hole](w,z,NSL::Slice(),i,l) 
                                    * corrPool_[NSL::Hubbard::Particle](x,y,NSL::Slice(),j,k)) 
                                    
                                    / Type(NumberTimeSources)));
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
