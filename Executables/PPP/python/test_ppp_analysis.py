#!/usr/bin/env python3
"""Tests of ppp_analysis.py on synthetic data, and (if given) on a reduced file of ppp_reduce.py.

    python test_ppp_analysis.py [reduced.h5 [reduced_binned.h5]]       # exit code 1 on failure

With a reduced file, the phi = 0 (non-interacting) correlators are checked: the addition correlator
of a conduction orbital and the removal correlator of a valence orbital decay exactly like
e^{-|eps| t}, and the free closed-shell ground state has <Q> = 0 and <S^2> = 0. With a second file,
reduced from the same chain with --bin B (and possibly --single), its binned correlators, equal-time
data and per-configuration probes are compared with those of the first (unbinned) file.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppp_analysis as pa  # noqa: E402


def check(results, name, value, tol):
    results.append((name, float(value), value < tol))


def main(argv):
    res = []
    rng = np.random.default_rng(1)

    # jackknife of a mean = standard error of the bins
    x = rng.normal(2.0, 1.0, (400, 3))
    full, err, _ = pa.jk_apply(lambda m: m, x)
    check(res, "jackknife error of a mean", np.max(np.abs(err - x.std(axis=0, ddof=1) / np.sqrt(len(x)))), 1e-12)

    # effective energy and exponential fit of an exact exponential
    delta, E, Nt = 0.25, 0.7, 64
    c = 3.0 * np.exp(-E * delta * np.arange(Nt))
    check(res, "effective energy of an exponential", np.max(np.abs(pa.effective_energy(c, delta) - E)), 1e-12)
    check(res, "exponential fit", abs(pa.fit_exp(c, 0.01 * c, delta, (4, 20))[0] - E), 1e-12)

    # GEVP: two states with non-orthogonal overlaps in a 3-operator basis
    Es = np.array([0.4, 0.9, 1.6])
    Zm = rng.normal(size=(3, 3))
    t = np.arange(Nt)
    C = np.einsum("an,bn,tn->tab", Zm, Zm, np.exp(-np.outer(t * delta, Es)))
    v = pa.gevp_vectors(C, 1, 4)
    proj = pa.gevp_project(C, v)
    Eeff = pa.effective_energy(proj.T, delta)
    check(res, "GEVP recovers the energies", np.max(np.abs(Eeff[:, 2:20] - Es[:, None])), 1e-9)

    # state_energies on noisy synthetic bins
    bins = C[None] * (1 + 0.01 * rng.normal(size=(200, Nt, 1, 1)))
    st = pa.state_energies(bins, delta, 1, 4, (4, 16))
    check(res, "state_energies within 3 sigma", np.max(np.abs(st["E"] - Es) / np.maximum(st["dE"], 1e-6)), 3.0)

    # consistency chi^2 of compatible estimates
    m, e, chi2, dof = pa.consistency([1.0, 1.1, 0.9], [0.1, 0.1, 0.1])
    check(res, "consistency: weighted mean", abs(m - 1.0), 1e-12)
    check(res, "consistency: chi2/dof", abs(chi2 - 1.0), 1e-12)

    # particle-hole symmetry: mu = 0 on a bipartite lattice only
    from types import SimpleNamespace as NS
    sub = np.array([0, 1, 0, 1])
    cases = [(NS(mu=0.0, sublattice=sub), True), (NS(mu=0.1, sublattice=sub), False),
             (NS(mu=0.0, sublattice=None), False)]
    check(res, "particle-hole symmetry test", sum(pa.particle_hole_symmetric(r) != want for r, want in cases), 0.5)

    # tail exponents: Student t with nu degrees of freedom has P(|x| > y) ~ y^-nu
    for nu in (1.5, 3.0):
        st = pa.tail_stats(rng.standard_t(nu, 400000), frac=0.002)
        check(res, f"tail exponent of Student t (nu = {nu}): relative error", abs(st["alpha"] / nu - 1), 0.15)
    st = pa.tail_stats(np.r_[rng.normal(size=999), 1e3])
    check(res, "tail_stats: largest deviation found", abs(st["argmax"] - 999), 0.5)

    # binned h5 reads of one time slice = time slice of the binned read
    import h5py
    with h5py.File("tail_test.h5", "w", driver="core", backing_store=False) as f:
        ds = f.create_dataset("x", data=rng.normal(size=(23, 7, 3, 3)) + 1j * rng.normal(size=(23, 7, 3, 3)))
        check(res, "binned read of one time slice", np.abs(pa._binned_dataset(ds, 4, 2, t=5)
                                                            - pa._binned_dataset(ds, 4, 2)[:, 5]).max(), 1e-14)

    # level slots, free two-body energies
    sl = pa._slots([(1.0, "--", "a"), (1.0, "++", "b"), (1.02, "-+", "c"), (1.5, "+-", "d")], tol=0.05)
    check(res, "level slots: degenerate levels side by side", float(sl != {1: (0, 3), 2: (1, 3), 0: (2, 3), 3: (0, 1)}), 0.5)
    eps = np.array([-1.0, -0.3, 0.3, 1.0])
    pe = pa.pair_energies([(2, 1), (3, 0), (3, 1)], eps, "exciton")
    pp = pa.pair_energies([(2, 2), (2, 3)], eps, "pp_singlet")
    M = np.array([[0, 1 / np.sqrt(2), -1 / np.sqrt(2)], [1, 0, 0]])
    check(res, "free pair energies (exciton, pp, combinations)",
          np.abs(pe - [0.6, 2.0, 1.3]).max() + np.abs(pp - [0.6, 1.3]).max()
          + np.abs(pa.combination_energies(M, pe) - [2.0, 0.6]).max(), 1e-14)

    if len(argv) > 1:
        ch = pa.Chain(argv[1])
        if ch.has_noninteracting:
            worst = 0.0
            for cls, idx in ch.classes.items():
                A, R = ch.noninteracting_onebody(cls)
                for k, o in enumerate(idx):
                    eps = ch.eps[o]
                    corr = A[:, k, k] if eps > 0 else R[:, k, k]
                    Eeff = pa.effective_energy(corr, ch.delta)
                    ok = corr[1:] > 1e-7 * corr[0]           # smaller values: round-off of the stored M^-1
                    worst = max(worst, np.max(np.abs(Eeff[ok] - abs(eps))))
            check(res, f"phi = 0: E_eff = |eps| ({ch.tag})", worst, 1e-9)
            import h5py
            with h5py.File(argv[1], "r") as f:
                G, ss, qq = (np.real(f[f"noninteracting/equaltime/{k}"][()])[None] for k in ("G", "ss", "qq"))
            obs = pa.equaltime_observables(G, ss, qq, ch.sublattice)
            check(res, "phi = 0: <Q> = 0", abs(obs["Q"][0]), 1e-9)
            # grand canonical at mu = 0: <S^2> = 3/2 sum_k f_k (1 - f_k), f_k = 1/(1 + e^{beta eps_k})
            fk = 1 / (1 + np.exp(ch.beta * ch.eps))
            check(res, "phi = 0: <S^2> = 3/2 sum f(1-f) (thermal)", abs(obs["S2"][0] / (1.5 * np.sum(fk * (1 - fk))) - 1), 1e-8)
        import h5py
        with h5py.File(argv[1], "r") as f:
            single = str(f.attrs.get("precision", "double")) == "single"
        tol_ph = 1e-6 if single else 1e-12
        A, R = ch.onebody(next(iter(ch.classes)), binsize=ch.bin if ch.bin > 1 else 10)
        check(res, f"interacting addition/removal symmetric ({ch.tag})",
              np.max(np.abs(A - np.swapaxes(A, -1, -2))) + np.max(np.abs(R - np.swapaxes(R, -1, -2))), 1e-14)

        # particle-hole identity per configuration (or bin; complex data): R of the partner orbitals = A^*
        worst, scale = 0.0, 0.0
        with h5py.File(argv[1], "r") as f:
            for cls, idx in ch.classes.items():
                cond = [o for o in idx if ch.eps[o] > 0]
                if not cond or ch.partner[cond[0]] < 0:
                    continue
                pc = ch.orbital_class[ch.partner[cond[0]]]
                pidx = list(ch.classes[pc])
                ia = [list(idx).index(o) for o in cond]
                ib = [pidx.index(ch.partner[o]) for o in cond]
                F, B = f[f"onebody/{cls}/forward"][:3], f[f"onebody/{cls}/backward"][:3]
                Fp, Bp = f[f"onebody/{pc}/forward"][:3], f[f"onebody/{pc}/backward"][:3]
                back = ch.Nt - np.arange(1, ch.Nt)
                Ac = 0.5 * (F[:, 1:] - B[:, back])[:, :, ia][:, :, :, ia]
                Rc = 0.5 * (-np.swapaxes(Bp[:, 1:], -1, -2) + np.swapaxes(Fp[:, back], -1, -2))[:, :, ib][:, :, :, ib]
                worst, scale = max(worst, np.abs(Rc - Ac.conj()).max()), max(scale, np.abs(Ac).max())
        check(res, f"per configuration: R(partner orbitals) = A^* ({ch.tag})", worst / scale, tol_ph)

    if len(argv) > 2:
        # the same chain reduced with --bin B: bins of the per-configuration data, probes
        c1, cb = pa.Chain(argv[1]), pa.Chain(argv[2])
        assert c1.bin == 1 and cb.bin > 1, "the second file must be the binned one"
        with h5py.File(argv[2], "r") as f:
            tol = 1e-6 if str(f.attrs.get("precision", "double")) == "single" else 1e-12
        worst = 0.0
        for cls in c1.classes:
            for X1, X2 in zip(c1.onebody(cls, 2 * cb.bin), cb.onebody(cls, 2 * cb.bin)):
                worst = max(worst, np.abs(X1 - X2).max() / np.abs(X1).max())
        check(res, f"--bin {cb.bin}: binned correlators = bins of the unbinned ones", worst, tol)
        worst = max(np.abs(a - b).max() / np.abs(a).max() for a, b in zip(c1.equaltime(cb.bin), cb.equaltime(cb.bin)))
        check(res, f"--bin {cb.bin}: binned equal-time data", worst, tol)
        worst = 0.0
        for cls, idx in cb.classes.items():
            for t in cb.probe_slices:
                for a in range(len(idx)):
                    x1, x2 = c1.onebody_element(cls, a, int(t)), cb.onebody_element(cls, a, int(t))
                    worst = max(worst, np.abs(x1 - x2).max() / max(np.abs(x1).max(), 1e-300))
        check(res, f"--bin {cb.bin}: per-configuration probes = A_aa(t)", worst, tol)
        check(res, f"--bin {cb.bin}: Monte Carlo history unchanged", float(np.abs(c1.S - cb.S).max()), 1e-12)

    failed = False
    for name, val, ok in res:
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:55s} {val:.1e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
