/** 
 * mostly copied from isle
 */

#ifndef NANOSYSTEMLIBRARY_HUBBARDFERMIACTION_HPP
#define NANOSYSTEMLIBRARY_HUBBARDFERMIACTION_HPP

#include "action.tpp"

namespace NSL::Action {
	

template<class Configuration>
class HubbardFermiAction : public ActionBase<Configuration> {
	public:
	/// Construct from individual parameters of HubbardFermiMatrix[Dia,Exp].
	HubbardFermiAction(const Lattice &lat, const double beta,
						const double muTilde, const std::int8_t sigmaKappa,
						const bool allowShortcut)
		: _hfm{lat, beta, muTilde, sigmaKappa},
			_kp{_hfm.K(Species::PARTICLE)},
			_kh{_hfm.K(Species::HOLE)},
			_shortcutForHoles{allowShortcut
							&& _internal::_holeShortcutPossible<BASIS>(
								lat.hopping(), muTilde, sigmaKappa)}
	{ }

	HubbardFermiAction(const HubbardFermiAction &other) = default;
	HubbardFermiAction &operator=(const HubbardFermiAction &other) = default;
	HubbardFermiAction(HubbardFermiAction &&other) = default;
	HubbardFermiAction &operator=(HubbardFermiAction &&other) = default;
	~HubbardFermiAction() override = default;

	complex<double> eval(Configuration & config) const override;

	Configuration force(Configuration & config) const override;
	
	Configuration grad(Configuration & config) const override;

	private:
	/// Stores all necessary parameters.
	const typename NSL::FermionMatrix::FermionMatrix<Type> _hfm;
	const typename _internal::KMatrixType<HOPPING>::type _kp;  ///< Matrix K for particles.
	const typename _internal::KMatrixType<HOPPING>::type _kh;  ///< Matrix K for holes.
	/// Can logdetM for holes be computed from logdetM from particles?
	const bool _shortcutForHoles;
};

}; // namespace NSL::Action

#endif  // NANOSYSTEMLIBRARY_HUBBARDFERMIACTION_HPP