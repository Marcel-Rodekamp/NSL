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

    # settings in tau -> time slices of runs at different delta (same physical times)
    bad = 0
    bad += (pa.tau_slice(1.5, 0.25), pa.tau_slice(1.5, 0.125), pa.tau_slice(1.5, 1 / 6)) != (6, 12, 9)
    try:
        pa.tau_slice(0.5, 1 / 3, "TAU0")
        bad += 1
    except ValueError as e:
        bad += "TAU0" not in str(e)
    bad += (pa.tau_window((2.0, 12.0), 0.25), pa.tau_window((2.0, 12.0), 0.125)) != ((8, 48), (16, 96))
    bad += pa.tau_window((0.5, 2.0), 1 / 3) != (2, 6)                       # inside the requested window

    class _Chain:                                                           # probe lookup needs bin, delta, probe_slices
        def __init__(self, b, d, p):
            self.bin, self.delta, self.probe_slices = b, d, np.array(p)
    bad += pa.probe_slice(_Chain(50, 0.125, [4, 8, 16, 32]), 2.0) != 16
    bad += pa.probe_slice(_Chain(50, 1 / 6, [4, 8, 16, 32]), 2.0) != 16    # nearest stored probe (tau = 2.67)
    bad += pa.probe_slice(_Chain(1, 1 / 6, []), 2.0) != 12
    check(res, "tau settings -> time slices (tau_slice, tau_window, probe_slice)", bad, 0.5)

    # intensive action: without fermions <S_gauge> = Nx Nt / 2, so s = 0 on average at every Nt
    worst = 0.0
    for Nt in (64, 128, 256):
        ch = _Chain(1, 32.0 / Nt, [])
        ch.Nx, ch.Nt, ch.beta = 4, Nt, 32.0
        ch.S_fermion = np.zeros(4000)
        ch.S = 0.5 * np.sum(rng.normal(size=(4000, 4 * Nt)) ** 2, axis=1)  # Gaussian field in its own units
        s, s_F = pa.action_densities(ch)
        worst = max(worst, abs(s.mean()) / (s.std() / np.sqrt(len(s))), np.abs(s_F).max())
    check(res, "action_densities: s = 0 without fermions (sigma)", worst, 4.0)

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

    # t = 0 contact term of old stage-2 files from a stage-1 file: delta_{v_i v_j} F(0)_{c_i c_j}, bin by bin
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p1, p2, p3 = (os.path.join(tmp, f"{k}.h5") for k in ("one", "two", "other"))
        F = {"A": rng.normal(size=(6, 4, 2, 2)), "B": rng.normal(size=(6, 4, 2, 2))}       # 6 stage-1 bins of 1
        for p, first in ((p1, 100), (p3, 200)):
            with h5py.File(p, "w") as f:
                f.attrs.update(dict(bin=1, n_bins=6, n_configs=6, sources=[0, 2]))
                f["config/id"], f["orbitals/U"] = np.arange(6) + first, np.eye(4)
                for cls, orbs in (("A", [0, 1]), ("B", [2, 3])):
                    f[f"onebody/{cls}/orbitals"], f[f"onebody/{cls}/forward"] = orbs, F[cls]
        pairs = np.array([[0, 2], [1, 2], [0, 3]])                       # conduction 0, 1 (class A), valence 2, 3
        with h5py.File(p2, "w") as f:                                    # 3 stage-2 bins of 2, without contact_terms
            f.attrs.update(dict(Nt=4, delta=0.5, bin=2, window_energy=1.0, sources=[0, 2]))
            f["orbitals/U"], f["orbitals/eps"], f["orbitals/partner"] = np.eye(4), [1.0, 0.5, -0.5, -1.0], [3, 2, 1, 0]
            f["orbitals/class_names"], f["orbitals/class"] = np.array([b"A", b"B"]), [0, 0, 1, 1]
            f["config/first_id"], f["exciton/s/pairs"] = [100, 102, 104], pairs
            f["exciton/s/connected"] = f["exciton/s/disconnected"] = np.zeros((3, 4, 3, 3))
            f["exciton/s/expectation"] = np.zeros((3, 2, 3))
        F0 = F["A"][:, 0].reshape(3, 2, 2, 2).mean(axis=1)[1:]           # stage-1 bins in pairs, first one discarded
        want = np.zeros((2, 3, 3))
        want[:, :2, :2], want[:, 2, 2] = F0, F0[:, 0, 0]                  # v_i = v_j: pairs 0, 1 (v = 2) and pair 2 (v = 3)
        X = pa.TwoBodyChain(p2, discard=2, onebody_path=p1).exciton("s", 1)[0]
        check(res, "t = 0 contact term from stage 1 (old stage-2 files)", np.abs(X[:, 0] - want).max() + np.abs(X[:, 1:]).max(), 1e-14)
        try:
            pa.TwoBodyChain(p2, onebody_path=p3)
            refused = 0.0
        except ValueError:
            refused = 1.0
        check(res, "stage-1 file of another chain refused", 1.0 - refused, 0.5)

    # bridge link: pseudo-bins of a ratio estimator reproduce the leave-one-out ratios exactly
    num, w = rng.normal(1.0, 1.0, (50, 4, 3)), rng.uniform(0.1, 2.0, 50)
    P = pa.reweight(num, w)
    loo = (num.sum(0)[None] - num) / (w.sum() - w)[:, None, None]
    check(res, "reweight: leave-one-out means = leave-one-out ratios", np.abs(pa.jackknife(P) - loo).max(), 1e-12)
    check(res, "reweight: mean = ratio (second order)", np.abs(P.mean(0) - num.sum(0) / w.sum()).max(), 1e-3)
    check(res, "reweight: weights 1 leave the bins unchanged", float(pa.reweight(num, np.ones(50)) is not num), 0.5)

    # bridge link: Lambda from G(b, b) = mean of the determinant ratios |det M_p(phi + c_k)|^2 / |det M_p(phi)|^2
    import ppp_ed
    import ppp_reduce
    Nx, Nt_b, delta_b, theta, b_sl = 4, 8, 0.25, np.pi / 2, 3
    Kc = np.diag(np.ones(Nx - 1), 1) + np.diag(np.ones(Nx - 1), -1)
    worst, worst_re = 0.0, 0.0
    for _ in range(3):
        phi = rng.normal(0, 1.0, (Nt_b, Nx))
        M = ppp_ed.fermion_matrix(Kc, phi, delta_b, "particle")
        Gbb = np.linalg.inv(M).reshape(Nt_b, Nx, Nt_b, Nx)[b_sl, :, b_sl, :]
        lam, dev = ppp_reduce.bridge_lambda(Gbb, theta)
        ld = np.linalg.slogdet(M)[1]
        ratios = []
        for i in range(Nx):
            for x in (1, -1):
                ph = phi.copy()
                ph[b_sl, i] += theta * x
                ratios.append(np.exp(2 * (np.linalg.slogdet(ppp_ed.fermion_matrix(Kc, ph, delta_b, "particle"))[1] - ld)))
        worst, worst_re = max(worst, abs(lam / np.mean(ratios) - 1)), max(worst_re, dev)
    check(res, "bridge_lambda = mean determinant ratio (NSL convention)", worst, 1e-10)
    check(res, "bridge: Re n = 1/2 (open chain, mu = 0)", worst_re, 1e-12)
    check(res, "bridge_settings: yml defaults", abs(ppp_reduce.bridge_settings({"HMC": {"bridge": None}})[0] - np.pi / 2)
          + ppp_reduce.bridge_settings({"HMC": {"bridge": None}})[1] + (ppp_reduce.bridge_settings({"HMC": {}}) is not None), 1e-15)

    # bridge link in the accessors: fake weighted stage-1 and stage-2 files of two chains (Run-wide ratios)
    with tempfile.TemporaryDirectory() as tmp:
        nbins, Ntf, nx = 8, 4, 2
        data = []
        for k in range(2):
            path1 = os.path.join(tmp, f"one/r_c0{k}.h5")
            path2 = os.path.join(tmp, f"two/r_c0{k}.h5")
            os.makedirs(os.path.dirname(path1), exist_ok=True)
            os.makedirs(os.path.dirname(path2), exist_ok=True)
            wk = rng.uniform(0.2, 2.0, nbins)
            eq = {q: rng.normal(size=(nbins, nx, nx)) for q in ("G", "ss", "qq")}
            Xc, Dc = rng.normal(size=(nbins, Ntf, 1, 1)), rng.normal(size=(nbins, Ntf, 1, 1))
            Ex, Pc = rng.normal(size=(nbins, 2, 1)), rng.normal(size=(nbins, Ntf, 1, 1))
            data.append((wk, eq, Xc, Dc, Ex, Pc))
            with h5py.File(path1, "w") as f:
                f.attrs.update(dict(name="r", Nt=Ntf, Nx=nx, beta=1.0, delta=0.25, mu=0.0, sources=[0], bin=1,
                                    n_bins=nbins, n_configs=nbins, input_yml_text="", bridge=1,
                                    bridge_theta=theta, bridge_slice=0))
                for q in ("K", "V", "positions", "V_eigenvalues"):
                    f[f"lattice/{q}"] = np.eye(nx)
                f["orbitals/U"], f["orbitals/eps"], f["orbitals/partner"] = np.eye(nx), [-1.0, 1.0], [1, 0]
                f["orbitals/class_names"], f["orbitals/class"] = np.array([b"all"]), [0, 0]
                f["onebody/all/orbitals"] = [0, 1]
                f["onebody/all/forward"] = f["onebody/all/backward"] = np.zeros((nbins, Ntf, nx, nx))
                for q in ("id", "S", "S_gauge", "acceptanceProbability", "detsign"):
                    f[f"config/{q}"] = np.zeros(nbins)
                f["config/field_modes"] = np.zeros((nbins, nx))
                f["config/bridge_lambda"] = 1 / wk
                f["weights"] = wk
                for q, v in eq.items():
                    f[f"equaltime/{q}"] = v
                f.create_group("checks")
            with h5py.File(path2, "w") as f:
                f.attrs.update(dict(Nt=Ntf, delta=0.25, bin=1, window_energy=1.0, sources=[0], contact_terms=1,
                                    bridge=1, bridge_theta=theta, bridge_slice=0))
                f["orbitals/U"], f["orbitals/eps"], f["orbitals/partner"] = np.eye(nx), [-1.0, 1.0], [1, 0]
                f["orbitals/class_names"], f["orbitals/class"] = np.array([b"all"]), [0, 0]
                f["config/first_id"], f["weights"] = np.arange(nbins), wk
                f["exciton/all/pairs"], f["pp_singlet/all/pairs"] = [[1, 0]], [[1, 1]]
                f["exciton/all/connected"], f["exciton/all/disconnected"] = Xc, Dc
                f["exciton/all/expectation"], f["pp_singlet/all/correlator"] = Ex, Pc
        W = np.concatenate([d[0] for d in data])
        ratio = lambda num: (num.sum(0) / W.sum())                       # noqa: E731
        run = pa.load_runs(os.path.join(tmp, "one"))["r"]
        ss = run.equaltime(1)[1]
        want = ratio(np.concatenate([d[1]["ss"] for d in data]))
        check(res, "bridge: Run.equaltime = run-wide ratio", np.abs(ss.mean(0) - want).max(), 1e-2 * np.abs(want).max())
        loo_ss = (np.concatenate([d[1]["ss"] for d in data]).sum(0)[None] - np.concatenate([d[1]["ss"] for d in data])) \
            / (W.sum() - W)[:, None, None]
        check(res, "bridge: Run.equaltime jackknife = leave-one-out ratios", np.abs(pa.jackknife(ss) - loo_ss).max(), 1e-12)
        chains2 = pa.load_two_body(os.path.join(tmp, "two"))["r"]
        Xall = np.concatenate([d[2] for d in data])
        Dall, Eall = np.concatenate([d[3] for d in data]), np.concatenate([d[4] for d in data])
        C = pa.exciton_bins(chains2, "all", "singlet", 1)
        ex = pa.reweight(Eall, W).mean(0)
        want_loo = ((Xall.sum(0)[None] - Xall) + 2 * (Dall.sum(0)[None] - Dall)) / (W.sum() - W)[:, None, None, None] \
            - 2 * np.outer(ex[0], ex[1])
        check(res, "bridge: exciton_bins jackknife = leave-one-out ratios", np.abs(pa.jackknife(C) - want_loo).max(), 1e-12)
        Pall = np.concatenate([d[5] for d in data])
        Pb = pa.pp_bins(chains2, "pp_singlet", "all", 1)
        check(res, "bridge: pp_bins jackknife = leave-one-out ratios",
              np.abs(pa.jackknife(Pb) - (Pall.sum(0)[None] - Pall) / (W.sum() - W)[:, None, None, None]).max(), 1e-12)
        summ = pa.bridge_summary(run.chains[0])
        check(res, "bridge_summary: N_eff/N", abs(summ["neff_frac"] - data[0][0].sum() ** 2 / (data[0][0] ** 2).sum() / nbins), 1e-12)

    # reweighted per-configuration series: mean = ratio, bin error = delta-method error of the ratio
    xs, ws = rng.normal(1.0, 0.5, 20000), rng.uniform(0.1, 2.0, 20000)
    u = pa.reweighted_series(xs, ws)
    R = np.sum(ws * xs) / np.sum(ws)
    xb, wb = pa.bin_mean(ws * xs, 100), pa.bin_mean(ws, 100)
    loo = (xb.sum() - xb) / (wb.sum() - wb)                       # leave-one-bin-out ratios
    err_u = pa.mean_error(u, 100)[1]
    check(res, "reweighted_series: mean = sum w x / sum w", abs(u.mean() - R), 1e-12)
    check(res, "reweighted_series: bin error = jackknife error of the ratio", abs(err_u / pa.jk_error(loo) - 1), 1e-2)
    check(res, "reweighted_series: unchanged without weights", np.abs(pa.reweighted_series(xs, None) - xs).max(), 1e-15)

    # run mean of chains: pooled mean (configuration counts) or run-wide ratio (sums of weights)
    chains_x = [rng.normal(1.0, 0.3, n) for n in (200, 350, 500)]
    chains_w = [rng.uniform(0.2, 2.0, len(x)) for x in chains_x]
    ms = [x.mean() for x in chains_x]
    es = [pa.mean_error(x, 10)[1] for x in chains_x]
    mean, err, chi2, pulls = pa.chain_agreement(ms, es, [len(x) for x in chains_x])
    allx = np.concatenate(chains_x)
    loo = [np.concatenate([x for j, x in enumerate(chains_x) if j != i]).mean() for i in range(3)]
    check(res, "chain_agreement: pooled mean and jackknife over chains",
          abs(mean - allx.mean()) + abs(err - pa.jk_error(np.array(loo))), 1e-12)
    check(res, "chain_agreement: chi2/dof and pulls",
          abs(chi2 - np.sum(((np.array(ms) - mean) / es) ** 2) / 2) + np.abs(pulls - (np.array(ms) - mean) / es).max(), 1e-12)
    rs = [pa.reweighted_series(x, w).mean() for x, w in zip(chains_x, chains_w)]
    ratio_all = sum((w * x).sum() for x, w in zip(chains_x, chains_w)) / sum(w.sum() for w in chains_w)
    check(res, "chain_agreement: reweighted chain means -> run-wide ratio",
          abs(pa.chain_agreement(rs, es, [w.sum() for w in chains_w])[0] - ratio_all), 1e-12)

    # sign fraction: plain counts, reweighted with the bridge link, jackknife over chains
    class FakeChain:
        def __init__(self, sign, lam):
            self.detsign, self.bridge_lambda, self.n = np.asarray(sign), lam, len(sign)
    signs = [rng.choice([1, -1, 0], 300, p=[0.6, 0.3, 0.1]) for _ in range(5)]
    lams = [rng.uniform(0.5, 5.0, 300) for _ in range(5)]
    f_plain, e_plain = pa.minus_fraction([FakeChain(s, None) for s in signs])
    f_bridge, _ = pa.minus_fraction([FakeChain(s, l) for s, l in zip(signs, lams)])
    alls, allw = np.concatenate(signs), 1 / np.concatenate(lams)
    per = np.array([np.mean(s[s != 0] < 0) * np.sum(s != 0) for s in signs]), np.array([np.sum(s != 0) for s in signs])
    jk = (per[0].sum() - per[0]) / (per[1].sum() - per[1])
    check(res, "minus_fraction: plain = count fraction, jackknife over chains",
          abs(f_plain - np.mean(alls[alls != 0] < 0)) + abs(e_plain - pa.jk_error(jk)), 1e-12)
    check(res, "minus_fraction: bridge = weighted fraction",
          abs(f_bridge - np.sum(allw[alls < 0]) / np.sum(allw[alls != 0])), 1e-12)

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
