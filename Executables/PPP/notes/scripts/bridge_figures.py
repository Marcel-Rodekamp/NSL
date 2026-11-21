#!/usr/bin/env python3
"""Figures for deck 7 (the bridge link), 4-site open Hubbard chain, U = 4, beta = 4, Nt = 16.

    cd Executables/PPP/notes/scripts
    for s in 1 2; do
        python bridge_proto.py run plain  100000 $s CHAINS/plain_$s.npz
        python bridge_proto.py run bridge 100000 $s CHAINS/bridge_$s.npz
    done
    python bridge_proto.py exact CHAINS/exact.npy
    python bridge_figures.py CHAINS            # writes ../images/bridge_*.png, prints the numbers in deck 7

The chains are not kept in git (4 x 5 MB, about 4 minutes each). Without CHAINS only the two figures that need
no Monte Carlo (bridge_ratios.png, bridge_node_cut.png) are made.
"""
import os
import sys

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "..", "images")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
import bridge_proto as bp  # noqa: E402
import ppp_analysis as pa  # noqa: E402

# validated categorical slots (dataviz reference palette): slot 1 = plain chain, slot 2 = bridge chain
PLAIN, BRIDGE = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID, FAINT = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
# ordinal blue ramp (steps 300, 450, 600) for theta = pi/4, pi/2, 3 pi/4
THETA_RAMP = ["#6da7ec", "#2a78d6", "#184f95"]
BIN = 500            # trajectories per bin
DISCARD = 2000       # thermalization (trajectories)
NAMES = [r"$S_\mathrm{AF}$", r"$S_\mathrm{CDW}$", r"$\langle s^2\rangle$"]

plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": TEXT, "legend.frameon": False})


def style(ax, grid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(True, color=GRID, lw=0.6, zorder=0)


def save(fig, name):
    out = os.path.join(IMAGES, name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(out))


# ---------------------------------------------------------------------------------------------------------------
def ratios():
    """R_{i,+-}(y) and Lambda for one site at theta = pi/2; the weight 1/Lambda against the rms y for three theta"""
    y = np.linspace(-1.5, 1.5, 601)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.7), dpi=200)
    th = np.pi / 2
    c, s = np.cos(th / 2), np.sin(th / 2)
    a.plot(y, (c - 2 * y * s) ** 2, color=PLAIN, lw=1.4, ls="--", label=r"$R_{i,+}$: shift $+\theta$")
    a.plot(y, (c + 2 * y * s) ** 2, color=PLAIN, lw=1.4, ls=":", label=r"$R_{i,-}$: shift $-\theta$")
    a.plot(y, c ** 2 + 4 * s ** 2 * y ** 2, color=BRIDGE, lw=2.2, label=r"mean $=\cos^2(\theta/2)+4\sin^2(\theta/2)\,y^2$")
    a.axhline(c ** 2, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    a.text(1.48, c ** 2 + 0.12, r"floor $\cos^2(\theta/2)=1/2$", ha="right", va="bottom", fontsize=10, color=MUTED)
    a.set_xlabel(r"$y_i(b)=\mathrm{Im}\,n_i(b)$")
    a.set_ylabel(r"bridge ratio")
    a.set_title(r"(a) one site, $\theta=\pi/2$", fontsize=11, loc="left")
    a.set_ylim(0, 6)
    a.legend(fontsize=9.5, loc="upper center")
    style(a)

    yb = np.linspace(0, 2.0, 401)
    for col, (th, lab) in zip(THETA_RAMP, [(np.pi / 4, r"\pi/4"), (np.pi / 2, r"\pi/2"), (3 * np.pi / 4, r"3\pi/4")]):
        c, s = np.cos(th / 2), np.sin(th / 2)
        b.plot(yb, 1 / (c ** 2 + 4 * s ** 2 * yb ** 2), color=col, lw=2, label=fr"$\theta={lab}$: $1/\Lambda\leq{1 / c ** 2:.2f}$")
    b.set_xlabel(r"$\bar y$: root mean square of $y_i(b)$ over the sites (large near a node)")
    b.set_ylabel(r"reweighting factor $1/\Lambda$")
    b.set_title(r"(b) weight of a configuration", fontsize=11, loc="left")
    b.set_ylim(0, 7)
    b.legend(fontsize=9.5, loc="upper right")
    style(b)
    fig.tight_layout()
    save(fig, "bridge_ratios.png")


# ---------------------------------------------------------------------------------------------------------------
def real_det(phi):
    """D(phi) = e^{-i sum phi / 2} det M_p(phi), real (mu = 0, bipartite)"""
    s, l = np.linalg.slogdet(bp.fermion_matrix(phi))
    return (s * np.exp(-0.5j * phi.sum())).real * np.exp(l)


def node_cut():
    """straight line phi(s) = (1 - s) phi_A + s phi_B between two Gaussian fields of opposite sign of D"""
    rng = np.random.default_rng(3)
    fields = [rng.normal(0, np.sqrt(bp.delta * bp.U), (bp.Nt, bp.Nx)) for _ in range(19)]
    A, B = fields[1], fields[18]
    path = lambda x: (1 - x) * A + x * B                                   # noqa: E731
    s0 = brentq(lambda x: real_det(path(x)), 0.45, 0.56, xtol=1e-15)
    near = s0 + np.concatenate([-np.geomspace(0.05, 1e-6, 120), np.geomspace(1e-6, 0.05, 120)])
    s = np.unique(np.concatenate([np.linspace(0, 1, 401), near]))
    det2, shifted, lam, saf = [], [], [], []
    for x in s:
        phi = path(x)
        _, Gt = bp.green(phi)
        R = bp.bridge_ratios(bp.density(Gt)[bp.BSLICE])
        d2 = real_det(phi) ** 2
        det2.append(d2)
        shifted.append(d2 * R)
        lam.append(R.mean())
        saf.append(bp.observables(Gt)[0])
    det2, shifted, lam, saf = map(np.array, (det2, shifted, lam, saf))
    norm = shifted.mean(axis=1).max()

    fig, (a, b) = plt.subplots(2, 1, figsize=(9.6, 6.6), dpi=200, sharex=True, gridspec_kw=dict(height_ratios=[1, 1.1]))
    for k in range(shifted.shape[1]):
        a.plot(s, shifted[:, k] / norm, color=FAINT, lw=0.8, label=r"$|\det M_p(\phi+c_k)|^2$, the $2N_x=8$ terms" if k == 0 else None)
    a.plot(s, det2 / norm, color=PLAIN, lw=2.2, label=r"$|\det M_p(\phi)|^2$: plain weight")
    a.plot(s, shifted.mean(axis=1) / norm, color=BRIDGE, lw=2.2, label=r"mean over $k$ $=|\det M_p(\phi)|^2\,\Lambda(\phi)$: bridge weight")
    a.axvline(s0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    a.text(s0 + 0.01, 0.93, r"node: $D(\phi)=0$", color=MUTED, fontsize=10, va="top")
    a.set_ylabel("fermion weight (scaled)")
    a.set_ylim(0, 1.05)
    a.legend(fontsize=9.5, loc="upper left", framealpha=0.92, facecolor="white", edgecolor="none")
    a.set_title(r"(a) the weight along a straight line through field space", fontsize=11, loc="left")
    style(a)

    b.semilogy(s, np.abs(saf), color=PLAIN, lw=2.2, label=r"$|S_\mathrm{AF}(\phi)|$: diverges like $1/D^2$")
    b.semilogy(s, np.abs(saf / lam), color=BRIDGE, lw=2.2, label=r"$|S_\mathrm{AF}(\phi)|/\Lambda(\phi)$: bounded")
    b.semilogy(s, 1 / lam, color=TEXT, lw=1.3, ls="--", label=r"$1/\Lambda(\phi)\leq 2$: vanishes at the node")
    b.axvline(s0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    b.set_xlabel(r"position $s$ on the line $\phi(s)=(1-s)\,\phi_A+s\,\phi_B$ ($\phi_A$, $\phi_B$: Gaussian fields, $D(\phi_A)>0>D(\phi_B)$)")
    b.set_ylabel("value on the configuration")
    b.set_ylim(1e-8, 1e9)
    b.legend(fontsize=9.5, loc="upper left")
    b.set_title(r"(b) per-configuration estimators along the same line", fontsize=11, loc="left")
    style(b)
    fig.tight_layout()
    save(fig, "bridge_node_cut.png")
    i0 = np.argmin(np.abs(s - s0))
    print(f"node cut: node at s = {s0:.6f}; closest point |s - s0| = {abs(s[i0] - s0):.1e}: "
          f"|det|^2/max {det2[i0] / det2.max():.1e}, S_AF {saf[i0]:.2e}, Lambda {lam[i0]:.2e}, "
          f"S_AF/Lambda {saf[i0] / lam[i0]:.3f}; bridge weight min/max on the line {shifted.mean(1).min() / norm:.3f}")


# ---------------------------------------------------------------------------------------------------------------
def load(chains):
    data = {}
    for kind in ("plain", "bridge"):
        for seed in (1, 2):
            d = np.load(os.path.join(chains, f"{kind}_{seed}.npz"))
            data[kind, seed] = {k: d[k][DISCARD:] for k in ("obs", "lam", "sign", "acc", "k")}
    return data, np.load(os.path.join(chains, "exact.npy"))


def estimate(obs, w, n=None):
    """reweighted ratio <w O>/<w> and its jackknife error from bins of BIN trajectories"""
    n = len(w) if n is None else n
    num, den = pa.bin_mean(w[:n] * obs[:n], BIN), pa.bin_mean(w[:n], BIN)
    est, err, _ = pa.jk_apply(lambda m: m[..., 0] / m[..., 1], np.stack([num, den], axis=1))
    return est, err


def weight(d, kind):
    return 1 / d["lam"] if kind == "bridge" else np.ones(len(d["lam"]))


def summary(data, exact):
    print("\nchains: estimate +- error (pull), sign changes, negative fraction, acceptance")
    for (kind, seed), d in data.items():
        w = weight(d, kind)
        parts = []
        for j in range(3):
            est, err = estimate(d["obs"][:, j], w)
            parts.append(f"{est:.4f}({err:.4f}) pull {(est - exact[j]) / err:+.1f}")
        sg = d["sign"]
        print(f"  {kind:6s} {seed}: " + "; ".join(parts)
              + f"; sign changes {int(np.sum(sg[1:] != sg[:-1]))}, - fraction {np.mean(sg < 0):.3f}, acc {d['acc'].mean():.3f}")
        st = pa.tail_stats(w * d["obs"][:, 0], frac=0.002)
        print(f"      S_AF numerator: Hill alpha {st['alpha']:.2f}, largest deviation {st['max_dev']:.3g} IQR")
        if kind == "bridge":
            neff = w.sum() ** 2 / (w ** 2).sum() / len(w)
            m, e = pa.mean_error(w, BIN)
            print(f"      1/Lambda in [{w.min():.4f}, {w.max():.4f}], <1/Lambda> = {m:.4f}({e:.4f}) "
                  f"vs 1/<s^2> = {1 / exact[2]:.4f}; N_eff/N = {neff:.3f}")
        else:
            m, e = pa.mean_error(d["lam"], BIN)
            print(f"      <Lambda> (= <s^2> at theta = pi/2, infinite variance here) = {m:.4f}({e:.4f}) vs {exact[2]:.4f}")
    print(f"  exact: S_AF {exact[0]:.4f}, S_CDW {exact[1]:.4f}, <s^2> {exact[2]:.4f}")


def weighted_survival(x, w):
    """(y, P(dev > y)) of the distribution with weights w, dev = |x - median| / IQR (weighted quantiles)"""
    order = np.argsort(x)
    cw = np.cumsum(w[order]) / w.sum()
    q = lambda p: x[order][np.searchsorted(cw, p)]                       # noqa: E731
    dev = np.abs(x - q(0.5)) / (q(0.75) - q(0.25))
    order = np.argsort(dev)
    tail = np.cumsum(w[order][::-1])[::-1] / w.sum()                      # weight with dev >= dev[order][j]
    return dev[order], tail


def tails(data):
    """the target distribution of S_AF has P(dev > y) ~ y^-3/2; the plain chain does not see that tail"""
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=200)
    o = np.concatenate([data["bridge", s]["obs"][:, 0] for s in (1, 2)])
    w = np.concatenate([1 / data["bridge", s]["lam"] for s in (1, 2)])
    y, p = weighted_survival(o, w)
    ax.loglog(y[y > 0.05], p[y > 0.05], color=BRIDGE, lw=2, ls="--",
              label=r"$S_\mathrm{AF}(\phi)$ in the target ensemble (bridge chains reweighted by $1/\Lambda$)")
    x = np.concatenate([data["plain", s]["obs"][:, 0] for s in (1, 2)])
    y, p = pa.tail_curve(x)
    ax.loglog(y[y > 0.05], p[y > 0.05], color=PLAIN, lw=2, label=r"$S_\mathrm{AF}(\phi)$ in the plain chains")
    x = o * w
    y, p = pa.tail_curve(x)
    ax.loglog(y[y > 0.05], p[y > 0.05], color=BRIDGE, lw=2, label=r"$S_\mathrm{AF}(\phi)/\Lambda(\phi)$ in the bridge chains")
    yy = np.geomspace(3, 1e9, 50)
    ax.loglog(yy, 0.4 * yy ** -1.5, color=TEXT, lw=1, ls=":", label=r"$0.4\,y^{-3/2}$")
    ax.set_xlabel(r"deviation $y=|X-\mathrm{median}\,X|\,/\,\mathrm{IQR}$")
    ax.set_ylabel(r"probability of a deviation $>y$")
    ax.set_ylim(1e-14, 1.5)
    ax.set_xlim(0.05, 1e11)
    ax.legend(fontsize=9.5, loc="lower left")
    style(ax)
    fig.tight_layout()
    save(fig, "bridge_tails.png")


def convergence(data, exact):
    ns = np.arange(4, 197, 4) * BIN
    fig = plt.figure(figsize=(11, 6.4), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes[1].sharey(axes[0])
    err_ax = fig.add_subplot(gs[1, :])
    for ax, kind, col in zip(axes, ("plain", "bridge"), (PLAIN, BRIDGE)):
        for seed, mk, ls in ((1, "o", "-"), (2, "s", "--")):
            d = data[kind, seed]
            w = weight(d, kind)
            e = np.array([estimate(d["obs"][:, 0], w, n) for n in ns])
            ax.fill_between(ns / 1000, e[:, 0] - e[:, 1], e[:, 0] + e[:, 1], color=col, alpha=0.18, lw=0)
            ax.plot(ns / 1000, e[:, 0], color=col, lw=1.6, ls=ls, label=f"seed {seed}")
            err_ax.loglog(ns, e[:, 1], color=col, lw=1.8, ls=ls, label=f"{kind}, seed {seed}")
        ax.axhline(exact[0], color=TEXT, lw=1, label="exact")
        ax.set_title(f"({'a' if kind == 'plain' else 'b'}) {kind} chain: running estimate $\\pm$ jackknife error",
                     fontsize=11, loc="left")
        ax.set_xlabel("trajectories (thousands)")
        ax.set_ylabel(r"$S_\mathrm{AF}$")
        ax.legend(fontsize=9.5, loc="upper right", ncol=3)
        style(ax)
    axes[0].set_ylim(1.4, 2.4)
    nn = np.geomspace(ns[0], ns[-1], 20)
    err_ax.loglog(nn, 0.022 * np.sqrt(ns[0] / nn), color=TEXT, lw=1, ls=":", label=r"$\propto N^{-1/2}$")
    err_ax.set_xlabel(r"trajectories $N$")
    err_ax.set_ylabel(r"jackknife error of $S_\mathrm{AF}$")
    err_ax.set_title("(c) jackknife error: the plain chains' error jumps when the chain sticks near a node",
                     fontsize=11, loc="left")
    err_ax.legend(fontsize=9.5, loc="lower left", ncol=5)
    style(err_ax)
    fig.tight_layout()
    save(fig, "bridge_convergence.png")


def weights_and_signs(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), dpi=200, gridspec_kw=dict(width_ratios=[1, 1.35]))
    w = np.concatenate([1 / data["bridge", s]["lam"] for s in (1, 2)])
    a.hist(w, bins=np.linspace(0, 2, 81), color=BRIDGE, edgecolor="white", lw=0.4, zorder=3)
    a.axvline(2, color=TEXT, lw=1, ls="--")
    a.text(1.97, a.get_ylim()[1] * 0.92, r"bound $1/\cos^2(\theta/2)=2$", ha="right", va="top", fontsize=10)
    neff = w.sum() ** 2 / (w ** 2).sum() / len(w)
    a.text(0.25, a.get_ylim()[1] * 0.92, f"$N_\\mathrm{{eff}}/N={neff:.2f}$", ha="left", va="top", fontsize=10)
    a.set_xlabel(r"reweighting factor $1/\Lambda$ (bridge chains)")
    a.set_ylabel("trajectories")
    a.set_title("(a) distribution of the weights", fontsize=11, loc="left")
    style(a)

    n = 20000
    for row, (kind, col) in enumerate((("plain", PLAIN), ("bridge", BRIDGE))):
        sg = data[kind, 1]["sign"][:n]
        neg = np.nonzero(sg < 0)[0]
        b.vlines(neg / 1000, row - 0.38, row + 0.38, color=col, lw=0.25)
        ch = int(np.sum(sg[1:] != sg[:-1]))
        b.text(n / 1000 * 1.01, row, f"{ch} sign changes", va="center", fontsize=9.5)
    b.set_yticks([0, 1], ["plain", "bridge"])
    b.set_ylim(-0.6, 1.6)
    b.set_xlim(0, n / 1000)
    b.set_xlabel("trajectory (thousands); a tick marks $D(\\phi)<0$")
    b.set_title(r"(b) sign of $D(\phi)$ along the chain (seed 1, first 20k)", fontsize=11, loc="left")
    style(b, grid=False)
    fig.tight_layout()
    save(fig, "bridge_weights_signs.png")


if __name__ == "__main__":
    os.makedirs(IMAGES, exist_ok=True)
    ratios()
    node_cut()
    if len(sys.argv) > 1:
        data, exact = load(sys.argv[1])
        summary(data, exact)
        tails(data)
        convergence(data, exact)
        weights_and_signs(data)
