#ifndef NSL_PPP_INTERACTION_TPP
#define NSL_PPP_INTERACTION_TPP

/*! \file PPP_Interaction.tpp
 *  \brief Interaction matrices of the Pariser-Parr-Pople (PPP) model.
 *
 *  The PPP Hamiltonian used throughout NSL is
 *  \f[
 *      H = -\sum_{xy,\sigma} \kappa_{xy} c^\dagger_{x\sigma} c_{y\sigma}
 *          + \frac{1}{2}\sum_{xy} V_{xy}\, q_x q_y ,
 *      \qquad q_x = n_{x\uparrow} + n_{x\downarrow} - 1 ,
 *  \f]
 *  with \f$V_{xx} = U\f$. Energies are measured in units of the hopping energy
 *  \f$t\f$ (the same units as `system: U` and `1/system: beta`), and site
 *  positions in units of `length unit`. The off-diagonal elements are
 *
 *   - Ohno:              \f$ V_{xy} = U / \sqrt{1 + (U\epsilon r_{xy}/e^2)^2} \f$
 *   - Mataga-Nishimoto:  \f$ V_{xy} = U / (1 + U\epsilon r_{xy}/e^2) \f$
 *   - on-site (Hubbard): \f$ V_{xy} = 0 \f$
 *   - explicit:          the full matrix is given in the yml file
 *
 *  where U and r are taken in eV and Angstrom, so that both forms approach the
 *  screened Coulomb tail \f$e^2/(\epsilon r)\f$ at large distances.
 *
 *  yml layout (all keys but `form` are optional, defaults in brackets):
 *  \code{.yml}
 *  PPP:
 *    form: ohno            # ohno | mataga-nishimoto | onsite | explicit
 *    hopping energy: 2.4   # t in eV                     [2.7]
 *    length unit: 1.4      # Angstrom per position unit  [1.4]
 *    dielectric: 2.0       # epsilon for x != y          [1.0]
 *    e2: 14.399645         # e^2/(4 pi eps0) in eV*A     [14.399645]
 *    V: [[...], ...]       # only for form: explicit, Nx x Nx in units of t
 *  \endcode
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include "../../Tensor.hpp"
#include "../../logger.hpp"

namespace NSL::PPP {

//! Coulomb constant \f$e^2/(4\pi\epsilon_0)\f$ in eV Angstrom
inline constexpr double e2_eV_Angstrom = 14.399645;

//! Functional form of the off-diagonal interaction
enum class Form { Ohno, MatagaNishimoto, OnSite, Explicit };

//! Parse a form name; case, '-', '_' and ' ' are ignored
inline Form formFromString(std::string name){
    std::string key;
    for (unsigned char c: name){
        if (c == '-' || c == '_' || c == ' ') { continue; }
        key += static_cast<char>(std::tolower(c));
    }
    if (key == "ohno")                                   { return Form::Ohno; }
    if (key == "mataganishimoto" || key == "mn")         { return Form::MatagaNishimoto; }
    if (key == "onsite" || key == "hubbard")             { return Form::OnSite; }
    if (key == "explicit" || key == "matrix")            { return Form::Explicit; }
    throw std::invalid_argument(
        "NSL::PPP: unknown interaction form '" + name
        + "' (expected ohno, mataga-nishimoto, onsite or explicit)"
    );
}

inline std::string to_string(Form form){
    switch (form){
        case Form::Ohno:            return "ohno";
        case Form::MatagaNishimoto: return "mataga-nishimoto";
        case Form::OnSite:          return "onsite";
        case Form::Explicit:        return "explicit";
    }
    return "unknown";
}

//! Parameters of the PPP interaction (energies in units of the hopping t)
struct Parameters {
    Form form = Form::Ohno;
    //! on-site interaction in units of t
    double U = 0.;
    //! hopping energy t in eV
    double hoppingEnergy = 2.7;
    //! Angstrom per unit of the site positions
    double lengthUnit = 1.4;
    //! dielectric constant applied to x != y
    double dielectric = 1.0;
    //! e^2/(4 pi eps_0) in eV Angstrom
    double e2 = e2_eV_Angstrom;
};

//! Off-diagonal interaction (units of t) between sites a distance r apart (units of lengthUnit)
inline double offDiagonal(const Parameters & p, double r){
    // U[eV] * eps * r[A] / e2
    const double x = p.U * p.hoppingEnergy * p.dielectric * p.lengthUnit * r / p.e2;
    switch (p.form){
        case Form::Ohno:            return p.U / std::sqrt(1. + x*x);
        case Form::MatagaNishimoto: return p.U / (1. + x);
        case Form::OnSite:          return 0.;
        case Form::Explicit:        break;
    }
    throw std::logic_error("NSL::PPP::offDiagonal: the explicit form has no distance dependence");
}

//! Interaction matrix from site positions
/*!
 *  \param positions Nx x d tensor of site positions (units of p.lengthUnit), any d
 *  \param p         interaction parameters (p.form must not be Explicit)
 *  \returns         Nx x Nx symmetric matrix in units of t with V_xx = U
 */
inline NSL::Tensor<double> interactionMatrix(const NSL::Tensor<double> & positions, const Parameters & p){
    torch::Tensor pos = torch::Tensor(positions).to(torch::kCPU).to(torch::kFloat64);
    if (pos.dim() != 2) {
        throw std::invalid_argument("NSL::PPP::interactionMatrix: positions must be a (Nx, d) tensor");
    }
    const NSL::size_t Nx = pos.size(0);
    auto P = pos.accessor<double,2>();

    NSL::Tensor<double> V(Nx, Nx);
    for (NSL::size_t x = 0; x < Nx; ++x){
        V(x, x) = p.U;
        for (NSL::size_t y = x+1; y < Nx; ++y){
            double r2 = 0.;
            for (NSL::size_t k = 0; k < pos.size(1); ++k){
                const double d = P[x][k] - P[y][k];
                r2 += d*d;
            }
            if (r2 == 0.){
                throw std::invalid_argument(fmt::format(
                    "NSL::PPP::interactionMatrix: sites {} and {} have identical positions", x, y
                ));
            }
            const double v = offDiagonal(p, std::sqrt(r2));
            V(x, y) = v;
            V(y, x) = v;
        }
    }
    return V;
}

//! Read the `PPP` block (and `system: U`) of an NSL yml file
inline Parameters parametersFromYAML(const YAML::Node & yml){
    if (!yml["PPP"] || !yml["PPP"]["form"]){
        throw std::invalid_argument("NSL::PPP: the yml file needs a `PPP: form:` entry (ohno, mataga-nishimoto, onsite or explicit)");
    }
    const YAML::Node ppp = yml["PPP"];
    Parameters p;
    p.form = formFromString(ppp["form"].as<std::string>());
    p.U    = yml["system"]["U"].as<double>();
    if (ppp["hopping energy"]) { p.hoppingEnergy = ppp["hopping energy"].as<double>(); }
    if (ppp["length unit"])    { p.lengthUnit    = ppp["length unit"].as<double>(); }
    if (ppp["dielectric"])     { p.dielectric    = ppp["dielectric"].as<double>(); }
    if (ppp["e2"])             { p.e2            = ppp["e2"].as<double>(); }
    return p;
}

//! Site positions (all given components, 2D or 3D) from `system: positions`
inline NSL::Tensor<double> positionsFromYAML(const YAML::Node & yml){
    const YAML::Node pos = yml["system"]["positions"];
    const NSL::size_t Nx = yml["system"]["nions"].as<NSL::size_t>();
    if (!pos || pos.size() != Nx){
        throw std::invalid_argument("NSL::PPP: `system: positions` must list `nions` positions");
    }
    const NSL::size_t dim = pos[0].size();
    NSL::Tensor<double> positions(Nx, dim);
    for (NSL::size_t x = 0; x < Nx; ++x){
        if (pos[x].size() != dim){
            throw std::invalid_argument("NSL::PPP: all positions must have the same number of components");
        }
        for (NSL::size_t k = 0; k < dim; ++k){
            positions(x, k) = pos[x][k].as<double>();
        }
    }
    return positions;
}

//! The interaction matrix V (units of t) described by an NSL yml file
inline NSL::Tensor<double> interactionMatrixFromYAML(const YAML::Node & yml){
    const Parameters p = parametersFromYAML(yml);
    if (p.form != Form::Explicit){
        return interactionMatrix(positionsFromYAML(yml), p);
    }

    const NSL::size_t Nx = yml["system"]["nions"].as<NSL::size_t>();
    const YAML::Node Vyml = yml["PPP"]["V"];
    if (!Vyml || Vyml.size() != Nx){
        throw std::invalid_argument("NSL::PPP: `form: explicit` requires `PPP: V:` as an nions x nions matrix");
    }
    NSL::Tensor<double> V(Nx, Nx);
    for (NSL::size_t x = 0; x < Nx; ++x){
        if (Vyml[x].size() != Nx){
            throw std::invalid_argument("NSL::PPP: `PPP: V:` must be an nions x nions matrix");
        }
        for (NSL::size_t y = 0; y < Nx; ++y){
            V(x, y) = Vyml[x][y].as<double>();
        }
        if (std::abs(Vyml[x][x].as<double>() - p.U) > 1e-12 * std::max(1., std::abs(p.U))){
            NSL::Logger::warn("NSL::PPP: explicit V[{0}][{0}] = {1} differs from system: U = {2}", x, Vyml[x][x].as<double>(), p.U);
        }
    }
    return V;
}

//! Log the interaction parameters and the spectrum of V
inline void logSummary(const Parameters & p, const NSL::Tensor<double> & V){
    torch::Tensor ev = torch::linalg_eigvalsh(torch::Tensor(V).to(torch::kCPU).to(torch::kFloat64), "L");
    const double lmin = ev.min().item<double>();
    const double lmax = ev.max().item<double>();
    NSL::Logger::info("PPP interaction: form={}, U={} t, t={} eV, length unit={} A, dielectric={}",
        to_string(p.form), p.U, p.hoppingEnergy, p.lengthUnit, p.dielectric);
    NSL::Logger::info("PPP interaction: eigenvalues of V in [{:.6}, {:.6}] t, condition number {:.4}",
        lmin, lmax, lmax/lmin);
}

} // namespace NSL::PPP

#endif // NSL_PPP_INTERACTION_TPP
