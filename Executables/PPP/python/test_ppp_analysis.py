#!/usr/bin/env python3
"""Tests of ppp_analysis.py on synthetic data, and (if given) on a reduced file of ppp_reduce.py.

    python test_ppp_analysis.py [reduced.h5]       # exit code 1 on failure

With a reduced file, the phi = 0 (non-interacting) correlators are checked: the addition correlator
of a conduction orbital and the removal correlator of a valence orbital decay exactly like
e^{-|eps| t}, and the free closed-shell ground state has <Q> = 0 and <S^2> = 0.
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
        A, R = ch.onebody(next(iter(ch.classes)), binsize=10)
        check(res, f"interacting addition/removal symmetric ({ch.tag})",
              np.max(np.abs(A - np.swapaxes(A, -1, -2))) + np.max(np.abs(R - np.swapaxes(R, -1, -2))), 1e-14)

        # particle-hole identity per configuration (complex data): R of the partner orbitals = A^*
        import h5py
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
        check(res, f"per configuration: R(partner orbitals) = A^* ({ch.tag})", worst / scale, 1e-12)

    failed = False
    for name, val, ok in res:
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:55s} {val:.1e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
