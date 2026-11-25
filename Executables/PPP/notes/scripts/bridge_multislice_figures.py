#!/usr/bin/env python3
"""Figures and numbers for the multi-slice bridge slides of deck 7.

    cd Executables/PPP/notes/scripts
    python bridge_multislice_figures.py --diag DIAG                  # perylene diagnostic -> bridge_ms_profile.png
    python bridge_multislice_figures.py --chains CHAINS_A CHAINS_B   # prototype chains   -> bridge_ms_tails.png

DIAG holds mn288_c25_outlier.npz and mn288_c25_random.npz from bridge_multislice_diag.py (Mataga-Nishimoto,
beta = 24, Nt = 288, theta = pi/2, b = 0, chain c25). Each CHAINS directory holds {plain,bridge,bridge_all}_{1,2}.npz
and exact.npy from bridge_proto.py (the same --beta and --nt for all of them); the numbers of the slides are printed.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bridge_figures as bf  # noqa: E402  (style, colours, estimate, BIN, DISCARD; imports ppp_analysis)

pa = bf.pa
# categorical slots 1-3 of the validated palette: plain, single-slice bridge, multi-slice bridge
ALL = "#1baf7a"
KINDS = ("plain", "bridge", "bridge_all")
COLOR = {"plain": bf.PLAIN, "bridge": bf.BRIDGE, "bridge_all": ALL}
DASH = {"plain": "-", "bridge": "--", "bridge_all": "-."}
LABEL = {"plain": "plain", "bridge": "single-slice bridge ($b=0$)", "bridge_all": "multi-slice bridge"}
NAMES = ["S_AF", "S_CDW", "<s^2>"]


# ---------------------------------------------------------------------------------------------------------------
def profile(diag):
    """(a) the bridge value of every slice, Lambda_t, on configurations of the outlier window and a typical one;
    (b) single-slice Lambda_0 against the all-slice Lambda_all on random configurations and the window"""
    win = np.load(os.path.join(diag, "mn288_c25_outlier.npz"), allow_pickle=False)
    rnd = np.load(os.path.join(diag, "mn288_c25_random.npz"), allow_pickle=False)
    beta, nt = 24.0, win["lam_t"].shape[1]
    tau = np.arange(nt) * beta / nt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.1), dpi=200, gridspec_kw=dict(width_ratios=[1.45, 1]))
    typical = int(np.argmin(np.abs(rnd["lam_all"] - np.median(rnd["lam_all"]))))
    a.semilogy(tau, rnd["lam_t"][typical], color=bf.FAINT, lw=1.2,
               label=fr"typical: configuration {int(rnd['id'][typical])}, $\Lambda_0={rnd['lam_b'][typical]:.2f}$, "
                     fr"$\Lambda_\mathrm{{all}}={rnd['lam_all'][typical]:.2f}$")
    ids = list(win["id"])
    for cfg, col in zip((1975, 1983, 1980), bf.THETA_RAMP):
        j = ids.index(cfg)
        a.semilogy(tau, win["lam_t"][j], color=col, lw=1.5,
                   label=fr"configuration {cfg}: $\Lambda_0={win['lam_b'][j]:.2f}$, "
                         fr"$\Lambda_\mathrm{{all}}={win['lam_all'][j]:.0f}$")
    a.axhline(0.5, color=bf.MUTED, lw=0.8, ls=(0, (4, 3)))
    a.text(beta * 0.99, 0.46, r"floor $\cos^2(\theta/2)=1/2$", ha="right", va="top", fontsize=9.5, color=bf.MUTED)
    a.annotate("bridge slice $b=0$", xy=(0, 0.62), xytext=(1.2, 8), fontsize=9.5, color=bf.TEXT,
               arrowprops=dict(arrowstyle="->", color=bf.MUTED, lw=0.8))
    a.set_xlim(0, beta)
    a.set_ylim(0.15, 1e8)
    a.set_xlabel(r"$\tau$")
    a.set_ylabel(r"$\Lambda_t$: bridge value at slice $t$")
    a.set_title(r"(a) $\Lambda_t$ along the slices (MN, $\beta=24$, $N_t=288$, chain c25)", fontsize=11, loc="left")
    a.legend(fontsize=8.8, loc="upper right")
    bf.style(a)

    b.loglog(rnd["lam_b"], rnd["lam_all"], "o", ms=3.2, color=bf.MUTED, alpha=0.75, label="200 random configurations")
    b.loglog(win["lam_b"], win["lam_all"], "s", ms=4.5, color=bf.THETA_RAMP[2],
             label="configurations 1975-1994 (outlier bin: 1980-1989)")
    lim = np.array([0.3, 1e4])
    b.loglog(lim, lim, color=bf.TEXT, lw=0.8, ls=":")
    b.text(9, 12.5, r"$\Lambda_\mathrm{all}=\Lambda_0$", rotation=33, fontsize=9.5, color=bf.MUTED)
    b.set_xlim(0.4, 30)
    b.set_ylim(0.4, 1e4)
    b.set_xlabel(r"$\Lambda_0$: single slice (the chain's weight)")
    b.set_ylabel(r"$\Lambda_\mathrm{all}$: mean over all slices")
    b.set_title(r"(b) both bridge values per configuration", fontsize=11, loc="left")
    b.legend(fontsize=8.8, loc="upper left")
    bf.style(b)
    fig.tight_layout()
    bf.save(fig, "bridge_ms_profile.png")

    # numbers for the slides
    l0, la, sx2 = rnd["lam_b"], rnd["lam_all"], rnd["sx2"]
    r = la / l0                                                       # reweights the sampled (w Lambda_0) ensemble
    e_all = lambda f: np.sum(r * f) / np.sum(r)                       # noqa: E731
    print("perylene diagnostic (MN, beta 24, Nt 288, chain c25)")
    for j, cfg in enumerate(win["id"]):
        print(f"  {int(cfg)}: Lambda_0 {win['lam_b'][j]:.4g}, Lambda_all {win['lam_all'][j]:.4g}, max_t Lambda_t "
              f"{win['lam_max'][j]:.4g} at tau {win['peak'][j] * beta / nt:.2f}, <s^2>_phi/Lambda_0 "
              f"{win['sx2'][j] / win['lam_b'][j]:.4g}, <s^2>_phi/Lambda_all {win['sx2'][j] / win['lam_all'][j]:.6f}")
    print(f"  random {len(l0)}: Lambda_0 median {np.median(l0):.3f} p99 {np.quantile(l0, .99):.3f} max {l0.max():.3f};"
          f" Lambda_all median {np.median(la):.3f} p99 {np.quantile(la, .99):.3f} max {la.max():.3f}")
    print(f"  <s^2>_phi/Lambda_0 in [{(sx2 / l0).min():.3f}, {(sx2 / l0).max():.3f}];"
          f" max |<s^2>_phi/Lambda_all - 1| = {np.abs(sx2 / la - 1).max():.1e}")
    print(f"  N_eff/N: single slice {np.sum(1 / l0) ** 2 / (len(l0) * np.sum(1 / l0 ** 2)):.3f} (sampled ensemble);"
          f" all slices, reweighted to w Lambda_all: {e_all(1 / la) ** 2 / e_all(1 / la ** 2):.3f}"
          f" (Kish N_eff/N of that reweighting {np.sum(r) ** 2 / (len(r) * np.sum(r ** 2)):.2f})")
    print(f"  peak slice quartiles (tau): {np.quantile(rnd['peak'], [.25, .5, .75]) * beta / nt}")


# ---------------------------------------------------------------------------------------------------------------
def load(chains):
    data = {}
    for kind in KINDS:
        for seed in (1, 2):
            path = os.path.join(chains, f"{kind}_{seed}.npz")
            if os.path.exists(path):
                d = np.load(path, allow_pickle=False)
                data[kind, seed] = {k: d[k][bf.DISCARD:] for k in ("obs", "lam", "lam_all", "sign", "acc")}
                params = (float(d["beta"]), int(d["Nt"]))
    return data, np.load(os.path.join(chains, "exact.npy")), params


def weight(d, kind):
    return {"plain": np.ones(len(d["lam"])), "bridge": 1 / d["lam"], "bridge_all": 1 / d["lam_all"]}[kind]


def longest_rejection_run(acc):
    run = best = 0
    for a in acc:
        run = 0 if a else run + 1
        best = max(best, run)
    return best


def summary(data, exact, params):
    beta, nt = params
    print(f"\n4-site chain, U = 4, beta = {beta:g}, Nt = {nt}; exact S_AF {exact[0]:.4f}, S_CDW {exact[1]:.4f}, "
          f"<s^2> {exact[2]:.4f}; bins of {bf.BIN}, first {bf.DISCARD} discarded")
    for kind in KINDS:
        seeds = [s for s in (1, 2) if (kind, s) in data]
        if not seeds:
            continue
        for seed in seeds + ["both"]:
            ds = [data[kind, s] for s in (seeds if seed == "both" else [seed])]
            obs = np.concatenate([d["obs"] for d in ds])
            w = np.concatenate([weight(d, kind) for d in ds])
            parts = []
            for j in range(3):
                est, err = bf.estimate(obs[:, j], w)
                parts.append(f"{NAMES[j]} {est:.4f}({err:.4f}) pull {(est - exact[j]) / err:+.1f}")
            line = f"  {kind:10s} {seed}: " + "; ".join(parts)
            if seed != "both":
                d = ds[0]
                sg = d["sign"]
                line += (f"\n      sign changes {int(np.sum(sg[1:] != sg[:-1]))}, - fraction (sampled) {np.mean(sg < 0):.3f},"
                         f" acceptance {d['acc'].mean():.3f}, longest rejection run {longest_rejection_run(d['acc'])}")
            print(line)
            for j in range(3):
                x = w * obs[:, j]
                if np.ptp(x) <= 1e-12 * np.abs(x).max():                 # <s^2>/Lambda_all at theta = pi/2
                    print(f"      {NAMES[j]} numerator w*O: constant, {x.mean():.15f} +- {np.ptp(x):.1e}")
                    continue
                st = pa.tail_stats(x, frac=0.002)
                print(f"      {NAMES[j]} numerator w*O: Hill alpha {st['alpha']:.2f}, largest deviation {st['max_dev']:.3g} IQR"
                      + (" (alpha undefined: the largest values repeat, the chain stuck)" if np.isinf(st['alpha']) else ""))
            if kind != "plain":
                neff = w.sum() ** 2 / (w ** 2).sum() / len(w)
                m, e = pa.mean_error(w, bf.BIN)
                print(f"      1/Lambda in [{w.min():.4f}, {w.max():.4f}], <1/Lambda> = {m:.4f}({e:.4f}) vs exact "
                      f"1/<s^2> = {1 / exact[2]:.4f} (pull {(m - 1 / exact[2]) / e:+.1f}); N_eff/N = {neff:.3f}")


def tails(sets):
    """survival function of the S_AF numerator (the per-configuration quantity that is averaged) per chain kind"""
    fig, axes = plt.subplots(1, len(sets), figsize=(5.6 * len(sets), 4.2), dpi=200, sharey=True, squeeze=False)
    for ax, (data, exact, (beta, nt)) in zip(axes[0], sets):
        for kind in KINDS:
            ds = [data[kind, s] for s in (1, 2) if (kind, s) in data]
            if not ds:
                continue
            x = np.concatenate([weight(d, kind) * d["obs"][:, 0] for d in ds])
            y, p = pa.tail_curve(x)
            keep = y > 0.05
            numer = {"plain": r"$S_\mathrm{AF}$", "bridge": r"$S_\mathrm{AF}/\Lambda_0$",
                     "bridge_all": r"$S_\mathrm{AF}/\Lambda_\mathrm{all}$"}[kind]
            ax.loglog(y[keep], p[keep], color=COLOR[kind], ls=DASH[kind], lw=2, label=f"{LABEL[kind]}: {numer}")
        yy = np.geomspace(3, 1e4, 40)
        ax.loglog(yy, 0.4 * yy ** -1.5, color=bf.TEXT, lw=1, ls=":", label=r"$0.4\,y^{-3/2}$ (target)")
        ax.set_xlabel(r"deviation $y=|X-\mathrm{median}\,X|\,/\,\mathrm{IQR}$")
        ax.set_title(fr"4-site chain, $\beta={beta:g}$, $N_t={nt}$", fontsize=11, loc="left")
        ax.set_xlim(0.05, 1e4)
        ax.set_ylim(3e-6, 1.5)
        bf.style(ax)
    axes[0][0].set_ylabel(r"fraction of trajectories with deviation $>y$")
    axes[0][0].legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    bf.save(fig, "bridge_ms_tails.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", help="directory with mn288_c25_{outlier,random}.npz")
    ap.add_argument("--chains", nargs="*", default=[], help="directories with prototype chains (one per beta)")
    args = ap.parse_args()
    if args.diag:
        profile(args.diag)
    sets = []
    for c in args.chains:
        data, exact, params = load(c)
        summary(data, exact, params)
        sets.append((data, exact, params))
    if sets:
        tails(sets)
