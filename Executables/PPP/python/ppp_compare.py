#!/usr/bin/env python3
"""Compare NSL PPP propagators with exact diagonalization (ppp_ed.py).

    python ppp_compare.py run.yml run.h5 run_ed.h5 [--plot run.pdf] [--check-configs 5]

1. deterministic check: for a few configurations NSL's measured invM[t,s],
   invM[s,t] and invM[t,t] are compared with a dense inverse of the fermion
   matrix built in numpy from the stored field;
2. statistical check: time-source averaged ensemble propagators (binned
   jackknife errors) are compared with the exact lattice propagators at the
   same Nt.
"""

import argparse
import os
import re
import sys

import numpy as np
import h5py
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppp_ed  # noqa: E402


def read_tensor(ds):
    return ds[()].reshape(tuple(ds.attrs["shape"]))


def config_ids(group):
    return sorted(int(k) for k in group.keys())


def source_columns(cfg_group, species):
    """{tsrc: invM[t, tsrc]} for the stored columns of one configuration"""
    out = {}
    g = cfg_group[f"propagator/{species}"]
    for key in g.keys():
        m = re.fullmatch(r"invM\[t,(\d+)\]", key)
        if m:
            out[int(m.group(1))] = read_tensor(g[key])
    return out


def averaged_column(cols, Nt):
    """sum_tsrc sign * invM[(tsrc+t)%Nt, tsrc] / #tsrc  (anti-periodic in time)"""
    acc = 0
    for s, col in cols.items():
        idx = (s + np.arange(Nt)) % Nt
        sign = np.where(s + np.arange(Nt) >= Nt, -1.0, 1.0)
        acc = acc + sign[:, None, None] * col[idx]
    return acc / len(cols)


def jackknife(data, binsize):
    """mean and jackknife error over axis 0 after binning"""
    nb = len(data) // binsize
    binned = data[: nb * binsize].reshape(nb, binsize, *data.shape[1:]).mean(axis=1)
    total = binned.sum(axis=0)
    jk = (total[None] - binned) / (nb - 1)
    mean = binned.mean(axis=0)
    err = np.sqrt((nb - 1) / nb * ((jk - jk.mean(axis=0)) ** 2).sum(axis=0))
    return mean, err


def deterministic_check(f, name, K, beta, Nt, ids, species_list=("particle", "hole")):
    delta = beta / Nt
    n = len(K)
    worst = 0.0
    for cid in ids:
        cfg = f[f"{name}/markovChain/{cid}"]
        phi = read_tensor(cfg["phi"])
        for sp in species_list:
            Minv = np.linalg.inv(ppp_ed.fermion_matrix(K, phi, delta, sp)).reshape(Nt, n, Nt, n)
            g = cfg[f"propagator/{sp}"]
            for key in g.keys():
                nsl = read_tensor(g[key])
                if key == "invM[t,t]":
                    ref = np.stack([Minv[t, :, t, :] for t in range(Nt)])
                elif (m := re.fullmatch(r"invM\[t,(\d+)\]", key)):
                    ref = Minv[:, :, int(m.group(1)), :]
                elif (m := re.fullmatch(r"invM\[(\d+),t\]", key)):
                    ref = Minv[int(m.group(1)), :, :, :].transpose(1, 0, 2)
                else:
                    continue
                worst = max(worst, np.abs(nsl - ref).max() / max(1e-300, np.abs(ref).max()))
    return worst


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("yml")
    p.add_argument("nsl_h5")
    p.add_argument("ed_h5")
    p.add_argument("--bin", type=int, default=0, help="bin size (default: ~ Ncfg/50)")
    p.add_argument("--discard", type=int, default=0, help="discard the first configurations")
    p.add_argument("--check-configs", type=int, default=5, help="configs for the deterministic check")
    p.add_argument("--plot", help="pdf with G_xx(t) and nearest-neighbour G_xy(t)")
    args = p.parse_args(argv)

    with open(args.yml) as fy:
        yml = yaml.safe_load(fy)
    name = yml["system"]["name"]
    K = ppp_ed.hopping_matrix(yml["system"])
    V = ppp_ed.interaction_matrix(yml)
    beta, Nt, n = float(yml["system"]["beta"]), int(yml["system"]["Nt"]), int(yml["system"]["nions"])

    with h5py.File(args.ed_h5, "r") as fe:
        G_ed = {sp: fe[f"lattice/{sp}"][()] for sp in ("particle", "hole")}
        assert np.allclose(fe["V"][()], V) and int(fe.attrs["Nt"]) == Nt and float(fe.attrs["beta"]) == beta, \
            "ED file does not match the yml file"

    with h5py.File(args.nsl_h5, "r") as f:
        V_nsl = read_tensor(f[f"{name}/Meta/PPP/V"])
        dV = np.abs(V_nsl - V).max()
        print(f"# {name}: n={n}, beta={beta}, Nt={Nt};  |V_NSL - V_python| = {dV:.2e}")
        ids = [i for i in config_ids(f[f"{name}/markovChain"]) if "propagator" in f[f"{name}/markovChain/{i}"]]
        ids = ids[args.discard:]
        acc = np.mean([f[f"{name}/markovChain/{i}/acceptanceRate"][()] for i in ids])
        print(f"# {len(ids)} measured configurations (ids {ids[0]}..{ids[-1]}), HMC acceptance {acc:.3f}")

        chk = ids[:: max(1, len(ids) // args.check_configs)][: args.check_configs]
        err = deterministic_check(f, name, K, beta, Nt, chk)
        print(f"deterministic check (configs {chk}): max relative |invM_NSL - inv(M_numpy)| = {err:.2e}")

        data = {sp: np.array([averaged_column(source_columns(f[f'{name}/markovChain/{i}'], sp), Nt) for i in ids])
                for sp in ("particle", "hole")}
        ntsrc = len(source_columns(f[f"{name}/markovChain/{ids[0]}"], "particle"))

    binsize = args.bin or max(1, len(ids) // 50)
    print(f"# binned jackknife: bin size {binsize}, {len(ids)//binsize} bins, {ntsrc} time sources per config")

    # pairs of sites at equal distance, for the correlated test
    pos = np.array(yml["system"]["positions"], float)
    dist = np.round(np.linalg.norm(pos[:, None] - pos[None], axis=-1), 6)
    classes = sorted(set(dist.ravel()))

    results = {}
    for sp in ("particle", "hole"):
        # NSL stores complex numbers; for real fields at mu = 0 the averages are real
        real = data[sp].real
        mean, err = jackknife(real, binsize)
        imag_mean, imag_err = jackknife(data[sp].imag, binsize)
        # entries fixed by symmetry on every configuration (e.g. G_xx(0) = 1/2 at half
        # filling) have round-off size errors: check them exactly, exclude them from z
        fixed = real.std(axis=0) < 1e-12 * max(1., np.abs(real).max())
        z = np.where(fixed, 0., (mean - G_ed[sp]) / np.where(fixed, 1., err))
        zf = z[~fixed]
        zi = (imag_mean / np.where(imag_err > 0, imag_err, np.inf))[~fixed]
        results[sp] = (mean, err, z)
        print(f"{sp:8s}: {fixed.sum()} configuration-independent entries, max|<G> - G_ED| there = "
              f"{np.abs(mean - G_ed[sp])[fixed].max() if fixed.any() else 0:.1e}")
        print(f"          other {zf.size} entries: <z^2> = {np.mean(zf**2):.3f}, max|z| = {np.abs(zf).max():.2f}, "
              f"|z|<1,2,3: {np.mean(np.abs(zf)<1):.3f} {np.mean(np.abs(zf)<2):.3f} {np.mean(np.abs(zf)<3):.3f} "
              f"(Gaussian: 0.683 0.954 0.997); Im part: max|<Im G>|/err = {np.abs(zi).max():.2f}; "
              f"rel. error of G_xx(Nt/2) = {np.mean(err[Nt//2].diagonal()/np.abs(mean[Nt//2].diagonal())):.1e}")

        # correlated chi^2 of G averaged over site pairs with equal distance, at <= ~64
        # observables (time slices are subsampled) so that the covariance is well estimated
        tsel = np.unique(np.linspace(0, Nt - 1, min(Nt, max(2, 64 // len(classes)))).round().astype(int))
        red = lambda A: np.stack([A[..., dist == c].mean(axis=-1) for c in classes], axis=-1)[..., tsel, :]
        X, Y = red(real).reshape(len(real), -1), red(G_ed[sp]).ravel()
        keep = X.std(axis=0) > 1e-12 * max(1., np.abs(X).max())
        X, Y = X[:, keep], Y[keep]
        cbin = max(1, min(binsize, len(X) // (8 * X.shape[1])))
        nb = len(X) // cbin
        B = X[: nb * cbin].reshape(nb, cbin, -1).mean(axis=1)
        w, U = np.linalg.eigh(np.cov(B, rowvar=False) / nb)
        ok = w > 1e-10 * w.max()
        diff = B.mean(axis=0) - Y
        chi2, dof = float(np.sum((U[:, ok].T @ diff) ** 2 / w[ok])), int(ok.sum())
        if nb > dof:
            from scipy import stats
            # Hotelling T^2: the covariance is estimated from the same nb bins
            pval = 1 - stats.f.cdf(chi2 * (nb - dof) / (dof * (nb - 1)), dof, nb - dof)
            print(f"          correlated chi^2 ({len(classes)} distance classes x {len(tsel)} t, {dof} observables, "
                  f"{nb} bins of {cbin}): chi^2/dof = {chi2/dof:.3f}, p = {pval:.3f}")
        else:
            print(f"          correlated chi^2 skipped: {dof} observables need more than {nb} bins")
    print("(z: entries of G are correlated, so the fractions fluctuate more than for independent data;"
          " the p-value of the correlated chi^2 is the reliable summary)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nn = [(x, y) for x in range(n) for y in range(x + 1, n) if K[x, y] != 0]
        pairs = [(0, 0), nn[0], (0, n // 2)]
        fig, axes = plt.subplots(2, len(pairs), figsize=(4.2 * len(pairs), 6.5), constrained_layout=True)
        ts = np.arange(Nt)
        for row, sp in enumerate(("particle", "hole")):
            mean, err, _ = results[sp]
            for col, (x, y) in enumerate(pairs):
                ax = axes[row, col]
                ax.plot(ts, G_ed[sp][:, x, y], "-", color="0.3", lw=1, label="exact (ED, same $N_t$)")
                ax.errorbar(ts, mean[:, x, y], err[:, x, y], fmt="o", ms=3, capsize=1.5, color="C0", label="NSL HMC")
                if x == y:
                    ax.set_yscale("log")
                ax.set_title(f"{sp}  $G_{{{x}{y}}}(t)$")
                ax.set_xlabel("$t$")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{name}: PPP ({yml['PPP']['form']}), $\\beta={beta:g}$, $N_t={Nt}$")
        fig.savefig(args.plot)
        print(f"# wrote {args.plot}")


if __name__ == "__main__":
    main()
