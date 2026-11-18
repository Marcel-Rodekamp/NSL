#!/usr/bin/env python3
"""Analysis of reduced PPP ensembles (the output of ppp_reduce.py), used by the notebook
Executables/PPP/analysis/ppp_ensemble_analysis.ipynb. Tests: test_ppp_analysis.py.

Data model
  Chain   one reduced file, i.e. one Markov chain (<run>_cNN.h5)
  Run     the chains of one input file (same physics, different seeds), e.g. perylene_ohno_b32_Nt128

Statistics
  Configurations are averaged in bins of `binsize` consecutive configurations of a chain; the bins of
  all chains of a run are the samples of a jackknife (jackknife, jk_error, jk_apply). Choose the bin
  size from the integrated autocorrelation times (tau_int) so that bins are independent. Files reduced
  with --bin B hold bins of B configurations already: binsize must then be a multiple of B, and
  `discard` is rounded up to whole bins of the file for the correlators (the history keeps it exactly).

Single-particle correlators (charge basis, orbital basis of ppp_reduce.py, per symmetry class)
  addition  A_ab(t) = <c_a(t) c^+_b(0)>   ~ sum_n e^{-(E_{N+1,n} - E_N) t}   (electron added)
  removal   R_ab(t) = <c^+_a(t) c_b(0)>   ~ sum_n e^{-(E_{N-1,n} - E_N) t}   (electron removed)
  Both use both stored time directions: A(t) = [F(t) - B(Nt-t)]/2, R(t) = [-B(t)^T + F(Nt-t)^T]/2,
  with F(t) = G(s+t, s) and B(t) = G(s, s+t) of ppp_reduce.py. Only the real, symmetric part is kept:
  the ensemble is invariant under phi -> -phi, which conjugates G, and the orbitals are real.
  At mu = 0 on a bipartite lattice, particle-hole conjugation gives, on every configuration,
  R_{v v'}(t) = A_{c c'}(t)^* for the valence orbitals v = (-1)^x c (checked to 1e-14 on the data):
  the removal spectrum of a class is the addition spectrum of the partner class.
  Energies are ratios E / kappa with the hopping kappa; imaginary time tau = delta * t in [0, beta)
  (t the time slice, delta = beta / Nt), in units of 1 / kappa.
"""

import glob
import os
import re
import sys

import numpy as np
import scipy.linalg as sla
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppp_autocorr  # noqa: E402

RUN_RE = re.compile(r"^(?P<run>.+)_c(?P<chain>\d+)$")

# categorical colours in fixed order (blue, orange, aqua, yellow, magenta, green, violet, red);
# a run or chain keeps its slot in every figure
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# ----------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------

def find_runs(data_dir):
    """{run: [reduced files]} for the files <run>_cNN.h5 (or <run>.h5) in data_dir, sorted"""
    runs = {}
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(data_dir), "*.h5"))):
        tag = os.path.splitext(os.path.basename(path))[0]
        m = RUN_RE.match(tag)
        runs.setdefault(m.group("run") if m else tag, []).append(path)
    return runs


def bin_mean(x, binsize):
    """average consecutive blocks of `binsize` along axis 0 (a remainder is dropped)"""
    nb = len(x) // binsize
    return np.asarray(x[: nb * binsize]).reshape(nb, binsize, *np.shape(x)[1:]).mean(axis=1)


def _binned_dataset(ds, binsize, start=0):
    nb = (ds.shape[0] - start) // binsize
    out = np.empty((nb, *ds.shape[1:]), dtype=np.result_type(ds.dtype, np.float64))
    for b in range(nb):
        out[b] = ds[start + b * binsize:start + (b + 1) * binsize].mean(axis=0)
    return out


def addition_removal(F, B):
    """addition and removal correlators (..., Nt, m, m) from the forward and backward ones; see module doc"""
    F, B = np.real(F), np.real(B)
    Nt, m = F.shape[-3], F.shape[-1]
    T = lambda X: np.swapaxes(X, -1, -2)                       # noqa: E731
    A, R = np.empty_like(F), np.empty_like(F)
    A[..., 0, :, :] = F[..., 0, :, :]
    R[..., 0, :, :] = np.eye(m) - T(F[..., 0, :, :])
    back = Nt - np.arange(1, Nt)
    A[..., 1:, :, :] = 0.5 * (F[..., 1:, :, :] - B[..., back, :, :])
    R[..., 1:, :, :] = 0.5 * (-T(B[..., 1:, :, :]) + T(F[..., back, :, :]))
    return 0.5 * (A + T(A)), 0.5 * (R + T(R))


class Chain:
    """One reduced file. Monte Carlo history is loaded at once; correlators on request, binned.
    The first `discard` configurations are left out everywhere (for files with bins of B configurations,
    the correlators drop the first ceil(discard / B) bins)."""

    def __init__(self, path, discard=0):
        self.path = path
        self.discard = d = int(discard)
        self.tag = os.path.splitext(os.path.basename(path))[0]
        m = RUN_RE.match(self.tag)
        self.run = m.group("run") if m else self.tag
        with h5py.File(path, "r") as f:
            a = f.attrs
            self.name, self.Nt, self.Nx = str(a["name"]), int(a["Nt"]), int(a["Nx"])
            self.beta, self.delta, self.mu = float(a["beta"]), float(a["delta"]), float(a["mu"])
            self.sources = np.array(a["sources"])
            self.bin = int(a.get("bin", 1))                  # configurations per stored bin (stage 1 --bin)
            self.skip = -(-d // self.bin)                    # bins dropped at the start (ceil)
            self.nbins_file = int(a.get("n_bins", a["n_configs"])) - self.skip
            self.yml_text = str(a["input_yml_text"])
            self.K, self.V = f["lattice/K"][()], f["lattice/V"][()]
            self.positions = f["lattice/positions"][()]
            self.V_eigenvalues = f["lattice/V_eigenvalues"][()]
            self.sublattice = f["lattice/sublattice"][()] if "lattice/sublattice" in f else None
            self.U, self.eps = f["orbitals/U"][()], f["orbitals/eps"][()]
            names = [s.decode() for s in f["orbitals/class_names"][()]]
            self.orbital_class = np.array([names[c] for c in f["orbitals/class"][()]])
            self.partner = f["orbitals/partner"][()]
            self.classes = {c: f[f"onebody/{c}/orbitals"][()] for c in f["onebody"]}
            cfg = f["config"]
            self.ids = cfg["id"][d:]
            self.S = cfg["S"][d:].real
            self.S_imag = cfg["S"][d:].imag
            self.S_gauge = cfg["S_gauge"][d:]
            self.S_fermion = self.S - self.S_gauge
            self.field_modes = cfg["field_modes"][d:]
            self.acceptance = cfg["acceptanceProbability"][d:]
            self.detsign = cfg["detsign"][d:]
            self.checks = dict(f["checks"].attrs)
            self.has_noninteracting = "noninteracting" in f
            c0 = next(iter(f["onebody"]))
            self.probe_slices = (f["onebody"].attrs["probe_slices"][()]
                                 if f"onebody/{c0}/addition_probe" in f else np.array([], int))
        self.n = len(self.ids)

    def rebin(self, binsize):
        """binsize (configurations) in units of the stored bins; it must be a multiple of the file's bin"""
        if binsize % self.bin:
            raise ValueError(f"{self.tag}: bin size {binsize} is not a multiple of the {self.bin} configurations "
                             f"per bin of the reduced file")
        return binsize // self.bin

    def nbins(self, binsize):
        return self.nbins_file // self.rebin(binsize)

    def __repr__(self):
        return f"Chain({self.tag}, {self.n} configurations)"

    def onebody(self, cls, binsize):
        """binned addition and removal correlators of a symmetry class, each (nbins, Nt, m, m)"""
        r = self.rebin(binsize)
        with h5py.File(self.path, "r") as f:
            F = _binned_dataset(f[f"onebody/{cls}/forward"], r, self.skip)
            B = _binned_dataset(f[f"onebody/{cls}/backward"], r, self.skip)
        return addition_removal(F, B)

    def onebody_element(self, cls, a, t, kind="addition"):
        """per-configuration A_aa(t) or R_aa(t) of the a-th orbital of the class (for tau_int); from the
        stored correlators if they are per configuration, otherwise from the probes (addition only, t one
        of probe_slices)"""
        if self.bin > 1:
            if kind != "addition" or t not in self.probe_slices:
                raise ValueError(f"{self.tag}: correlators are stored in bins of {self.bin}; per configuration only "
                                 f"the addition probes at t = {list(self.probe_slices)} exist")
            k = int(np.nonzero(self.probe_slices == t)[0][0])
            with h5py.File(self.path, "r") as f:
                return np.asarray(f[f"onebody/{cls}/addition_probe"][self.discard:, k, a], float)
        with h5py.File(self.path, "r") as f:
            F, B, d = f[f"onebody/{cls}/forward"], f[f"onebody/{cls}/backward"], self.discard
            if kind == "addition":
                x = 0.5 * (F[d:, t, a, a] - B[d:, self.Nt - t, a, a]) if t else F[d:, 0, a, a]
            else:
                x = 0.5 * (-B[d:, t, a, a] + F[d:, self.Nt - t, a, a]) if t else 1 - F[d:, 0, a, a]
        return np.real(x)

    def noninteracting_onebody(self, cls):
        """addition and removal correlators for phi = 0, each (Nt, m, m)"""
        with h5py.File(self.path, "r") as f:
            g = f[f"noninteracting/onebody/{cls}"]
            return addition_removal(g["forward"][()], g["backward"][()])

    def equaltime(self, binsize):
        """binned equal-time G = <c c^+>, <s_x s_y>, <q_x q_y> (real parts), each (nbins, Nx, Nx)"""
        r = self.rebin(binsize)
        with h5py.File(self.path, "r") as f:
            return tuple(np.real(_binned_dataset(f[f"equaltime/{k}"], r, self.skip)) for k in ("G", "ss", "qq"))


class Run:
    """The chains of one run; bins of all chains are concatenated (chain order) for the jackknife."""

    def __init__(self, name, paths, discard=0):
        self.name = name
        self.chains = [Chain(p, discard) for p in paths]
        c0 = self.chains[0]
        for c in self.chains[1:]:
            assert (c.Nt, c.Nx, c.beta) == (c0.Nt, c0.Nx, c0.beta), f"{c.tag} differs from {c0.tag}"
            assert np.allclose(c.U, c0.U), f"{c.tag}: orbital basis differs from {c0.tag}"
        for k in ("Nt", "Nx", "beta", "delta", "mu", "U", "eps", "orbital_class", "partner", "classes",
                  "positions", "sublattice", "K", "V", "V_eigenvalues", "yml_text", "has_noninteracting"):
            setattr(self, k, getattr(c0, k))
        self.label = name

    def __repr__(self):
        return f"Run({self.name}: {len(self.chains)} chains, {sum(c.n for c in self.chains)} configurations)"

    @property
    def n(self):
        return sum(c.n for c in self.chains)

    def onebody(self, cls, binsize):
        """binned (A, R) of all chains, concatenated: (nbins_total, Nt, m, m) each"""
        parts = [c.onebody(cls, binsize) for c in self.chains]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def equaltime(self, binsize):
        parts = [c.equaltime(binsize) for c in self.chains]
        return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))

    def chain_of_bins(self, binsize):
        return np.concatenate([np.full(c.nbins(binsize), i) for i, c in enumerate(self.chains)])


def load_runs(data_dir, order=None, discard=0):
    """{run name: Run}; `order` (list of run names) fixes the order (and colour slots) of the runs;
    `discard`: configurations dropped at the start of every chain (a number, or {run: number})"""
    found = find_runs(data_dir)
    names = [n for n in (order or []) if n in found] + sorted(n for n in found if n not in (order or []))
    disc = discard if isinstance(discard, dict) else {n: discard for n in names}
    return {n: Run(n, found[n], disc.get(n, 0)) for n in names}


# ----------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------

def jackknife(bins):
    """leave-one-out means of the bins (axis 0)"""
    bins = np.asarray(bins)
    return (bins.sum(axis=0)[None] - bins) / (len(bins) - 1)


def jk_error(samples):
    s = np.asarray(samples)
    n = len(s)
    return np.sqrt((n - 1) / n * np.sum(np.abs(s - s.mean(axis=0)) ** 2, axis=0))


def jk_apply(f, bins):
    """(f(mean), jackknife error, jackknife samples) of a function of the bin average"""
    full = f(np.mean(bins, axis=0))
    samples = np.array([f(s) for s in jackknife(bins)])
    return full, jk_error(samples), samples


def mean_error(x, binsize=1):
    """mean and error of a series (error from bins of binsize)"""
    b = bin_mean(np.asarray(x), binsize)
    return b.mean(axis=0), b.std(axis=0, ddof=1) / np.sqrt(len(b))


def tau_int(x):
    """integrated autocorrelation time (Gamma method, U. Wolff) and its error, in units of configurations"""
    tau, dtau, _ = ppp_autocorr.gamma_method(np.asarray(x, float))
    return tau, dtau


def tau_int_binned(x, binsizes):
    """tau_int estimated from the variance of bin averages, for each bin size (catches long excursions)"""
    return np.array([ppp_autocorr.tau_binned(np.asarray(x, float), b) for b in binsizes])


def consistency(means, errors):
    """weighted mean, its error, chi^2 / dof and dof of independent estimates (e.g. of different chains)"""
    means, errors = np.asarray(means, float), np.asarray(errors, float)
    w = 1 / errors ** 2
    m = np.sum(w * means) / np.sum(w)
    dof = len(means) - 1
    chi2 = np.sum(w * (means - m) ** 2)
    return m, 1 / np.sqrt(np.sum(w)), (chi2 / dof if dof else np.nan), dof


def sign_census(chain):
    """counts of determinant signs and number of sign changes along the chain (computed configurations only)"""
    s = chain.detsign[chain.detsign != 0]
    return dict(chain=chain.tag, computed=len(s), plus=int(np.sum(s > 0)), minus=int(np.sum(s < 0)),
                changes=int(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0)


# ----------------------------------------------------------------------------------------------
# spectroscopy
# ----------------------------------------------------------------------------------------------

def effective_energy(c, delta):
    """E_eff(t) = log(c(t) / c(t+1)) / delta along the last axis (time); NaN where undefined"""
    c = np.asarray(c, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(c[..., :-1] / c[..., 1:]) / delta


def gevp_vectors(C, t0, td):
    """generalized eigenvectors of C(td) v = lambda C(t0) v for a symmetric correlator matrix C (Nt, m, m),
    sorted by decreasing lambda (lowest energy first); normalized so that v^T C(t0) v = 1"""
    w, v = sla.eigh(0.5 * (C[td] + C[td].T), 0.5 * (C[t0] + C[t0].T))
    return v[:, np.argsort(w)[::-1]]


def gevp_project(C, v):
    """diagonal correlators v_n^T C(t) v_n of the GEVP states: (..., Nt, m) -> (..., Nt, nstates)"""
    return np.einsum("an,...tab,bn->...tn", v, C, v)


def orbital_selection(run, cls, kind):
    """indices (within the class block) of the orbitals used for addition (eps > 0) or removal (eps < 0)"""
    eps = run.eps[run.classes[cls]]
    return np.nonzero(eps > 0)[0] if kind == "addition" else np.nonzero(eps < 0)[0]


def fit_exp(c, err, delta, window):
    """fit c(t) = Z exp(-E t delta) on t in [window[0], window[1]] (uncorrelated weights); returns (E, Z)"""
    t = np.arange(window[0], window[1] + 1)
    y, s = np.asarray(c)[t], np.asarray(err)[t]
    mask = (y > 0) & np.isfinite(y) & (s > 0)
    if mask.sum() < 2:
        return np.nan, np.nan
    t, y, s = t[mask], y[mask], s[mask]
    # linear least squares for log c with weights (c / err)^2
    w = (y / s) ** 2
    Amat = np.vstack([np.ones_like(t, dtype=float), -t * delta]).T
    coef = np.linalg.lstsq(Amat * np.sqrt(w)[:, None], np.log(y) * np.sqrt(w), rcond=None)[0]
    return coef[1], np.exp(coef[0])


def signal_window(c, err, window, max_rel_err):
    """`window` cut at the first time slice where c <= 0 or err/c > max_rel_err; None if < 3 slices remain"""
    t1, t2 = window
    end = t1 - 1
    for t in range(t1, min(t2, len(c) - 1) + 1):
        if not (c[t] > 0 and err[t] < max_rel_err * c[t]):
            break
        end = t
    return (t1, end) if end - t1 >= 2 else None


def state_energies(C_bins, delta, t0, td, window, select=None, max_rel_err=0.25, overrides=None):
    """GEVP energies of a correlator matrix: C_bins (nbins, Nt, m, m), optional orbital selection.

    The GEVP vectors come from the mean (t0, td); each jackknife sample is projected on them and its
    diagonal correlators fitted with one exponential. The fit window of state n is overrides[n] if given,
    otherwise `window` cut where the relative error of the state's correlator exceeds max_rel_err
    (signal_window). Returns dict with E, dE and the windows (per state), the projected mean correlators
    and their errors (Nt, nstates), the effective energies with errors (nstates, Nt-1), and the vectors."""
    if select is not None:
        if len(select) == 0:
            return None
        C_bins = C_bins[:, :, select][:, :, :, select]
    mean = C_bins.mean(axis=0)
    v = gevp_vectors(mean, t0, td)
    proj = gevp_project(C_bins, v)                       # (nbins, Nt, nstates)
    jk = jackknife(proj)
    cm, ce = gevp_project(mean, v), jk_error(jk)
    ns = v.shape[1]
    windows = [(overrides or {}).get(n) or signal_window(cm[:, n], ce[:, n], window, max_rel_err) for n in range(ns)]
    E, dE = np.full(ns, np.nan), np.full(ns, np.nan)
    for n, w in enumerate(windows):
        if w is None:
            continue
        E[n] = fit_exp(cm[:, n], ce[:, n], delta, w)[0]
        dE[n] = jk_error(np.array([fit_exp(s[:, n], ce[:, n], delta, w)[0] for s in jk]))
    return dict(E=E, dE=dE, windows=windows, corr=cm, corr_err=ce, vectors=v,
                Eeff=effective_energy(cm.T, delta), Eeff_err=jk_error(effective_energy(np.swapaxes(jk, 1, 2), delta)))


# ----------------------------------------------------------------------------------------------
# equal-time observables
# ----------------------------------------------------------------------------------------------

def equaltime_observables(G, ss, qq, sublattice=None):
    """per-bin derived observables from the binned equal-time data (nbins, Nx, Nx) -> dict of (nbins, ...)

    n_x = <n_up + n_down>, Q = sum_x <q_x>, S2 = <S_total^2> = 3/4 sum_xy <s_x s_y>, m2_x = <s_x^2>
    (local moment), d_x = <n_up n_down> = (n_x - m2_x) / 2; with a sublattice: the staggered spin and
    charge structure factors S_AF = 1/Nx sum_xy e_x e_y <s_x s_y>, S_CDW = 1/Nx sum_xy e_x e_y <q_x q_y>
    (e = +-1 the sublattice sign)."""
    Nx = G.shape[-1]
    n = 2 * (1 - np.diagonal(G, axis1=-2, axis2=-1))
    m2 = np.diagonal(ss, axis1=-2, axis2=-1)
    out = dict(n=n, Q=(n - 1).sum(axis=-1), S2=0.75 * ss.sum(axis=(-2, -1)), m2=m2, d=(n - m2) / 2,
               ss=ss, qq=qq)
    if sublattice is not None:
        e = np.asarray(sublattice)
        out["S_AF"] = np.einsum("x,...xy,y->...", e, ss, e) / Nx
        out["S_CDW"] = np.einsum("x,...xy,y->...", e, qq, e) / Nx
    return out


def distance_shells(positions, decimals=3):
    """distance of every site pair and the sorted distinct distances (shells)"""
    p = np.asarray(positions, float)
    r = np.round(np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1), decimals)
    return r, np.unique(r)


def shell_average(M, r, shells):
    """average of M[..., x, y] over the site pairs of each distance shell -> (..., nshells)"""
    return np.stack([M[..., r == d].mean(axis=-1) for d in shells], axis=-1)


# ----------------------------------------------------------------------------------------------
# two-body correlators (stage 2, ppp_reduce2.py)
# ----------------------------------------------------------------------------------------------

class TwoBodyChain:
    """One stage-2 file: exciton (connected, disconnected, expectation) and pp correlators per sector."""

    def __init__(self, path, discard=0):
        self.path = path
        self.tag = os.path.splitext(os.path.basename(path))[0]
        m = RUN_RE.match(self.tag)
        self.run = m.group("run") if m else self.tag
        with h5py.File(path, "r") as f:
            a = f.attrs
            self.Nt, self.delta, self.bin = int(a["Nt"]), float(a["delta"]), int(a["bin"])
            self.window_energy = float(a["window_energy"])
            self.eps, self.partner = f["orbitals/eps"][()], f["orbitals/partner"][()]
            names = [s.decode() for s in f["orbitals/class_names"][()]]
            self.orbital_class = np.array([names[c] for c in f["orbitals/class"][()]])
            self.pairs = {(ch, s): f[f"{ch}/{s}/pairs"][()] for ch in ("exciton", "pp_singlet", "pp_triplet")
                          if ch in f for s in f[ch]}
            self.first_id = f["config/first_id"][()]
        self.skip = -(-int(discard) // self.bin)            # bins dropped at the start (ceil)
        self.nbins = len(self.first_id) - self.skip

    def rebin(self, binsize):
        """binsize (configurations) in units of the stored bins; it must be a multiple of the file's bin"""
        if binsize % self.bin:
            raise ValueError(f"{self.tag}: bin size {binsize} is not a multiple of the {self.bin} configurations "
                             f"per bin of the stage-2 file")
        return binsize // self.bin

    def _read(self, node, rebin):
        with h5py.File(self.path, "r") as f:
            return _binned_dataset(f[node], rebin, self.skip)

    def exciton(self, sector, rebin=1):
        """binned (connected X, disconnected D, expectation (2, n)) of an exciton sector (complex)"""
        return tuple(self._read(f"exciton/{sector}/{k}", rebin) for k in ("connected", "disconnected", "expectation"))

    def pp(self, kind, sector, rebin=1):
        return self._read(f"{kind}/{sector}/correlator", rebin)


def load_two_body(data_dir, order=None, discard=0):
    """{run: [TwoBodyChain]} for the stage-2 files in data_dir (same naming as stage 1)"""
    found = find_runs(data_dir)
    names = [n for n in (order or []) if n in found] + sorted(n for n in found if n not in (order or []))
    disc = discard if isinstance(discard, dict) else {n: discard for n in names}
    return {n: [TwoBodyChain(p, disc.get(n, 0)) for p in found[n]] for n in names}


def parity_transform(pairs, partner, channel):
    """rows = operator combinations of definite particle-hole parity C (spin-symmetric conjugation, = (-1)^I in the
    Hubbard model): pair i = (c, v), partner pair i' = (partner(v), partner(c)), combinations (O_i + s O_i')/sqrt(2);
    C = -s for the singlet, +s for the triplet (a self-partner pair, v = partner(c): C = -1 singlet, +1 triplet).
    Returns {+1: M_plus, -1: M_minus} (each (n_C, n)), and a label per row."""
    index = {(int(c), int(v)): i for i, (c, v) in enumerate(pairs)}
    rows, labels, done = {+1: [], -1: []}, {+1: [], -1: []}, set()
    n = len(pairs)
    sgn_C = (lambda s: -s) if channel == "singlet" else (lambda s: s)        # noqa: E731
    for i, (c, v) in enumerate(pairs):
        if i in done:
            continue
        j = index[(int(partner[v]), int(partner[c]))]
        done |= {i, j}
        if j == i:
            r = np.zeros(n); r[i] = 1.0
            C = -1 if channel == "singlet" else +1
            rows[C].append(r); labels[C].append(f"({c},{v})")
            continue
        for s in (+1, -1):
            r = np.zeros(n); r[i], r[j] = 1 / np.sqrt(2), s / np.sqrt(2)
            rows[sgn_C(s)].append(r); labels[sgn_C(s)].append(f"({c},{v}){'+' if s > 0 else '-'}({pairs[j][0]},{pairs[j][1]})")
    return {C: np.array(rows[C]).reshape(-1, n) for C in (+1, -1)}, labels


def exciton_bins(chains, sector, channel, binsize):
    """binned exciton correlator matrices of a run (all chains): triplet = X, singlet = X + 2 (D - <O><O^+>),
    with the ensemble averages <O>, <O^+> (vacuum subtraction; zero in expectation outside the vacuum sector).
    Real symmetric part; returns (nbins, Nt, n, n)."""
    parts = [c.exciton(sector, c.rebin(binsize)) for c in chains]
    X = np.concatenate([p[0] for p in parts])
    if channel == "triplet":
        C = X
    else:
        D = np.concatenate([p[1] for p in parts])
        ex = np.concatenate([p[2] for p in parts]).mean(axis=0)
        C = X + 2 * (D - np.outer(ex[0], ex[1])[None, None])
    C = np.real(C)
    return 0.5 * (C + np.swapaxes(C, -1, -2))


def pp_bins(chains, kind, sector, binsize):
    C = np.real(np.concatenate([c.pp(kind, sector, c.rebin(binsize)) for c in chains]))
    return 0.5 * (C + np.swapaxes(C, -1, -2))
