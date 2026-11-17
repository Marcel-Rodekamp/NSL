#ifndef NSL_EXACT_FOURIER_ACCELERATION_TPP
#define NSL_EXACT_FOURIER_ACCELERATION_TPP

#include <cmath>
#include <stdexcept>
#include <string>

#include "../integrator.tpp"
#include "Configuration/Configuration.tpp"
#include "typePromotion.hpp"

namespace NSL::Integrator {

//! Molecular dynamics with exact Fourier acceleration (EFA).
/*!
 * For an action with a time-local quadratic (gauge) part
 *  \f[ S(\phi) = S_0(\phi) + S_1(\phi), \qquad
 *      S_0 = \frac{1}{2}\sum_t \phi_t^T A\, \phi_t , \f]
 * where A is a symmetric positive-definite \f$N_x\times N_x\f$ matrix
 * (\f$A = (\delta V)^{-1}\f$ for NSL::Action::PPPGaugeAction,
 *  \f$A = 1/(\delta U)\f$ for NSL::Action::HubbardGaugeAction),
 * the molecular dynamics uses the mass matrix A,
 *  \f[ H = \frac{1}{2}\sum_t p_t^T A^{-1} p_t + S(\phi). \f]
 * \f$H_0 = \frac{1}{2}p^TA^{-1}p + S_0\f$ is integrated exactly: every mode of
 * the gauge action oscillates with unit frequency,
 *  \f[ \phi(\tau) = \phi\cos\tau + A^{-1}p\,\sin\tau ,\qquad
 *      p(\tau) = p\cos\tau - A\phi\,\sin\tau , \f]
 * and only the remainder \f$S_1 = S - S_0\f$ (the fermion action) is integrated with
 * kicks, in the symmetric splitting
 * \f$R(\epsilon/2)\,K(\epsilon)\,R(\epsilon/2)\f$ (one force evaluation per step).
 * The gauge action therefore produces no integration error for any step size, and
 * its stiff and slow modes (\f$\omega_k = (\delta\lambda_k)^{-1/2}\f$ without
 * acceleration) all move at the same speed. A trajectory length of \f$\pi/2\f$
 * decorrelates the Gaussian part completely in a single trajectory.
 *
 * Reference: J. Ostmeyer and P. Buividovich, "Minimal autocorrelation in Hybrid
 * Monte Carlo simulations using exact Fourier acceleration", arXiv:2404.09723.
 *
 * The integrator works with the full action (as Leapfrog does) and removes the
 * force of \f$S_0\f$ itself, so the dynamics are exact (reversible and
 * area-preserving) for any A; A only has to match the gauge action for the method
 * to be efficient.
 *
 * As LeapfrogRealForce, only the real part of the field is evolved with the real
 * part of the force; a constant imaginary part (offset) is kept fixed. The momenta
 * are real.
 *
 * The HMC classes call `sampleMomentum` (\f$p_t\sim\mathcal{N}(0,A)\f$) and
 * `kineticEnergy` (\f$\frac{1}{2}\sum_t p_t^TA^{-1}p_t\f$) through
 * NSL::Integrator::sampleMomentum / NSL::Integrator::kineticEnergy.
 * */
template<typename ... ActionTermTypes>
class ExactFourierAcceleration: Integrator<ActionTermTypes...> {
    public:

    //! \param action           full action (gauge + fermion)
    //! \param A                Nx x Nx quadratic form of the gauge action per time slice
    //! \param trajectoryLength molecular dynamics time; pi/2 is optimal for the Gaussian part
    //! \param numberSteps      number of force evaluations (kicks) per trajectory
    //! \param backward         integrate backward in time
    //! \param fieldName        configuration key of the field
    ExactFourierAcceleration(const NSL::Action::Action<ActionTermTypes...> & action,
                             const torch::Tensor & A,
                             const double & trajectoryLength,
                             const NSL::size_t & numberSteps,
                             bool backward = false,
                             const std::string & fieldName = "phi"):
        Integrator<ActionTermTypes...>(action),
        trajLength_(trajectoryLength),
        numSteps_(numberSteps),
        stepSize_(trajectoryLength/static_cast<double>(numberSteps)),
        key_(fieldName)
    {
        if (numberSteps < 1) {
            throw std::invalid_argument("NSL::Integrator::ExactFourierAcceleration: numberSteps must be at least 1");
        }
        if (backward){
            stepSize_ *= -1;
        }
        setMassMatrix_(A);
    }

    //! Takes A from a gauge action that provides `invVtilde()` (NSL::Action::PPPGaugeAction).
    template<typename GaugeActionType>
        requires requires(const GaugeActionType & gauge){ gauge.invVtilde(); }
    ExactFourierAcceleration(const NSL::Action::Action<ActionTermTypes...> & action,
                             const GaugeActionType & gauge,
                             const double & trajectoryLength,
                             const NSL::size_t & numberSteps,
                             bool backward = false,
                             const std::string & fieldName = "phi"):
        ExactFourierAcceleration(action, torch::Tensor(gauge.invVtilde()), trajectoryLength, numberSteps, backward, fieldName)
    {}

    //! Integrate from (q_, p_) for one trajectory.
    template<NSL::Concept::isNumber TensorType>
    std::tuple<NSL::Configuration<TensorType>, NSL::Configuration<TensorType> > operator()(
        const NSL::Configuration<TensorType> & q_, const NSL::Configuration<TensorType> & p_
    ){
        checkConfiguration_(q_);

        // deep copy to generate a new configuration
        NSL::Configuration<TensorType> q (q_,true);
        NSL::Configuration<TensorType> p (p_,true);

        const torch::Tensor phi = q.at(key_);
        const bool isComplex = phi.is_complex();
        const auto options = realOptions_(phi);
        const torch::Tensor A    = A_.to(options);
        const torch::Tensor Ainv = Ainv_.to(options);

        // real part (evolved), imaginary part (fixed), real momentum
        torch::Tensor x = (isComplex ? torch::real(phi) : phi).clone();
        const torch::Tensor y = isComplex ? torch::imag(phi).clone() : torch::Tensor();
        const torch::Tensor pp = p.at(key_);
        torch::Tensor P = (isComplex ? torch::real(pp) : pp).clone();

        auto field = [&](){
            return NSL::Tensor<TensorType>(isComplex ? torch::complex(x, y) : x.clone());
        };

        // exact flow of H_0 for a time h
        auto rotate = [&](double h){
            const double c = std::cos(h), s = std::sin(h);
            torch::Tensor xNew = c*x + s*P.matmul(Ainv);
            P = c*P - s*x.matmul(A);
            x = std::move(xNew);
        };

        // kick with the force of S - S_0 for a time h
        auto kick = [&](double h){
            q[key_] = field();
            torch::Tensor F = this->action_.force(q)[key_];
            if (isComplex) { F = torch::real(F); }
            // the force of S_0 is -A x; remove it from the full force
            P = P + h*(F + x.matmul(A));
        };

        rotate(0.5*stepSize_);
        for(NSL::size_t n = 0; n < numSteps_; ++n){
            kick(stepSize_);
            rotate( (n+1 < numSteps_) ? stepSize_ : 0.5*stepSize_ );
        }

        q[key_] = field();
        p[key_] = NSL::Tensor<TensorType>(isComplex ? torch::complex(P, torch::zeros_like(P)) : P);
        return {q,p};
    }

    //! Momentum with the distribution exp(-kineticEnergy): p_t ~ N(0, A), real
    template<NSL::Concept::isNumber TensorType>
    NSL::Configuration<TensorType> sampleMomentum(const NSL::Configuration<TensorType> & q){
        checkConfiguration_(q);
        const torch::Tensor phi = q.at(key_);
        const auto options = realOptions_(phi);
        torch::Tensor P = torch::randn(phi.sizes(), options).matmul(sqrtA_.to(options));
        return NSL::Configuration<TensorType>{
            {key_, NSL::Tensor<TensorType>(phi.is_complex() ? torch::complex(P, torch::zeros_like(P)) : P)}
        };
    }

    //! \f$\frac{1}{2}\sum_t p_t^T A^{-1} p_t\f$ (real part of p)
    template<NSL::Concept::isNumber TensorType>
    TensorType kineticEnergy(const NSL::Configuration<TensorType> & p){
        checkConfiguration_(p);
        const torch::Tensor pp = p.at(key_);
        const torch::Tensor P = pp.is_complex() ? torch::real(pp) : pp;
        const double K = 0.5 * (P * P.matmul(Ainv_.to(realOptions_(pp)))).sum().template item<double>();
        return static_cast<TensorType>(K);
    }

    //! quadratic form A of the gauge action (double, CPU)
    const torch::Tensor & massMatrix() const { return A_; }
    double trajectoryLength() const { return trajLength_; }
    NSL::size_t numberSteps() const { return numSteps_; }

    //! Number of MD steps per trajectory; the trajectory length stays fixed (used by NSL::MCMC::Autotune)
    void setNumberSteps(NSL::size_t numberSteps){
        if (numberSteps < 1) {
            throw std::invalid_argument("NSL::Integrator: the number of MD steps must be at least 1");
        }
        const double sign = (stepSize_ < 0) ? -1. : 1.;
        numSteps_ = numberSteps;
        stepSize_ = sign * trajLength_ / static_cast<double>(numberSteps);
    }

    protected:
        double trajLength_;
        NSL::size_t numSteps_;
        double stepSize_;
        std::string key_;
        // A, A^{-1}, A^{1/2}: double precision on the CPU, converted on use
        torch::Tensor A_;
        torch::Tensor Ainv_;
        torch::Tensor sqrtA_;

    void setMassMatrix_(const torch::Tensor & Ain){
        torch::Tensor A = Ain.to(torch::kCPU);
        if (A.is_complex()) {
            if (torch::imag(A).abs().max().template item<double>() > 1e-14 * std::max(1., A.abs().max().template item<double>())) {
                throw std::invalid_argument("NSL::Integrator::ExactFourierAcceleration: the mass matrix must be real");
            }
            A = torch::real(A);
        }
        A = A.to(torch::kFloat64).contiguous();
        if (A.dim() != 2 || A.size(0) != A.size(1)) {
            throw std::invalid_argument("NSL::Integrator::ExactFourierAcceleration: the mass matrix must be a square (Nx, Nx) matrix");
        }
        const double scale = std::max(1., A.abs().max().template item<double>());
        if ((A - A.transpose(0,1)).abs().max().template item<double>() > 1e-12 * scale) {
            throw std::invalid_argument("NSL::Integrator::ExactFourierAcceleration: the mass matrix must be symmetric");
        }
        auto [lambda, Q] = torch::linalg_eigh(A, "L");
        if (!(lambda.min().template item<double>() > 0.)) {
            throw std::invalid_argument("NSL::Integrator::ExactFourierAcceleration: the mass matrix must be positive definite");
        }
        const torch::Tensor Qt = Q.transpose(0,1);
        torch::Tensor Ainv  = Q.matmul(torch::diag(1./lambda)).matmul(Qt);
        torch::Tensor sqrtA = Q.matmul(torch::diag(torch::sqrt(lambda))).matmul(Qt);
        A_     = 0.5 * (A + A.transpose(0,1));
        Ainv_  = 0.5 * (Ainv  + Ainv.transpose(0,1));
        sqrtA_ = 0.5 * (sqrtA + sqrtA.transpose(0,1));
    }

    template<NSL::Concept::isNumber TensorType>
    void checkConfiguration_(const NSL::Configuration<TensorType> & config) const {
        if (config.size() != 1 || !config.contains(key_)) {
            throw std::invalid_argument(fmt::format(
                "NSL::Integrator::ExactFourierAcceleration: expects a configuration with the single field '{}'", key_
            ));
        }
        const torch::Tensor field = config.at(key_);
        if (field.dim() < 1 || field.size(-1) != A_.size(0)) {
            throw std::invalid_argument(fmt::format(
                "NSL::Integrator::ExactFourierAcceleration: the field has {} sites per time slice, the mass matrix is {}x{}",
                field.dim() < 1 ? 0 : field.size(-1), A_.size(0), A_.size(0)
            ));
        }
    }

    static torch::TensorOptions realOptions_(const torch::Tensor & field){
        const auto dtype = field.is_complex() ? c10::toRealValueType(field.scalar_type()) : field.scalar_type();
        return torch::TensorOptions().dtype(dtype).device(field.device());
    }
};

}
#endif //NSL_EXACT_FOURIER_ACCELERATION_TPP
