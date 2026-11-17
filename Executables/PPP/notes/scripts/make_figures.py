#!/usr/bin/env python3
"""Figures for the notes in Executables/PPP/notes.

    python scripts/make_figures.py          (run from Executables/PPP/notes)

1. converts the PDF figures in ../figures that the slides use to PNG (pdftoppm);
2. gauge_modes_perylene.png: frequencies of the gauge-action normal modes of perylene
   with unit mass, omega_k = (delta lambda_k)^(-1/2), against exact Fourier acceleration
   (omega = 1 for every mode).
3. perylene_modes.png: the eigenvectors of V with the largest (k = 0, uniform charge mode)
   and the smallest eigenvalue (k = 19, staggered) drawn on the molecule.
"""

import os
import subprocess
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.dirname(HERE)
PPP = os.path.dirname(NOTES)
sys.path.insert(0, os.path.join(PPP, "python"))
import ppp_ed  # noqa: E402

IMAGES = os.path.join(NOTES, "images")
PDFS = ["naphthalene_b20", "efa_autocorrelation_perylene", "efa_autocorrelation_naphthalene"]

# validated categorical slots used in figures/efa_autocorrelation_*.pdf:
# slot 1 = leapfrog T=1, slot 4 = EFA T=pi/2
LEAPFROG, EFA = "#2a78d6", "#eda100"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# diverging pair for signed eigenvector components
POSITIVE, NEGATIVE = "#2a78d6", "#e34948"


def perylene_spectrum():
    """V of yml/perylene.yml, eigenpairs ordered by decreasing eigenvalue (k = 0: largest)."""
    with open(os.path.join(PPP, "yml", "perylene.yml")) as fh:
        yml = yaml.safe_load(fh)
    lam, Q = np.linalg.eigh(ppp_ed.interaction_matrix(yml))
    order = np.argsort(lam)[::-1]
    return yml, lam[order], Q[:, order]


def convert_pdfs():
    for name in PDFS:
        src = os.path.join(PPP, "figures", f"{name}.pdf")
        subprocess.run(["pdftoppm", "-png", "-r", "160", "-singlefile", src, os.path.join(IMAGES, name)], check=True)


def gauge_modes():
    yml, lam, _ = perylene_spectrum()
    delta = float(yml["system"]["beta"]) / int(yml["system"]["Nt"])
    omega = 1.0 / np.sqrt(delta * lam)
    k = np.arange(len(lam))

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    ax.plot(k, omega, "o", color=LEAPFROG, ms=6, mec="white", mew=1.0, zorder=3,
            label=r"unit mass (leapfrog): $\omega_k = (\delta\lambda_k)^{-1/2}$")
    ax.axhline(1.0, color=EFA, lw=2, zorder=2, label=r"mass matrix $(\delta V)^{-1}$ (EFA): $\omega_k = 1$")
    ax.annotate(fr"$k=0$, uniform charge mode: $\omega_0 = {omega[0]:.2f}$", (k[0], omega[0]),
                xytext=(1.2, 0.18), textcoords="data", fontsize=10, color=TEXT,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate(fr"$k={k[-1]}$, staggered (stiffest): $\omega = {omega[-1]:.2f}$", (k[-1], omega[-1]),
                xytext=(18.7, 1.22), textcoords="data", fontsize=10, color=TEXT, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, relpos=(0.97, 1.0)))
    ax.set_xlabel(r"mode $k$ (eigenvectors of $V$, ordered by decreasing eigenvalue $\lambda_k$)")
    ax.set_ylabel(r"frequency $\omega_k$")
    ax.set_xticks(k[::2])
    ax.set_ylim(0, 2.5)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    out = os.path.join(IMAGES, "gauge_modes_perylene.png")
    fig.savefig(out)
    print(f"{out}: lambda(V) = {lam[-1]:.3f} ... {lam[0]:.3f}, omega = {omega[0]:.3f} ... {omega[-1]:.3f}, "
          f"cos(omega_0) = {np.cos(omega[0]):.3f}, cos(omega_max) = {np.cos(omega[-1]):.3f}")


def mode_shapes():
    yml, lam, Q = perylene_spectrum()
    delta = float(yml["system"]["beta"]) / int(yml["system"]["Nt"])
    pos = np.array(yml["system"]["positions"], dtype=float)[:, :2]
    bonds = yml["system"]["adjacency"]
    n = len(lam)
    uniform = 1.0 / np.sqrt(n)

    # sublattice signs, for the overlap of the last mode with the staggered pattern
    sign = np.zeros(n)
    sign[0] = 1
    todo = [0]
    while todo:
        i = todo.pop()
        for a, b in bonds:
            j = b if a == i else a if b == i else None
            if j is not None and sign[j] == 0:
                sign[j] = -sign[i]
                todo.append(j)

    v0 = Q[:, 0] * np.sign(Q[:, 0].sum())
    vs = Q[:, -1] * np.sign(Q[:, -1] @ sign)
    w0 = (v0 @ np.ones(n)) ** 2 / n
    ws = (vs @ sign) ** 2 / n

    plt.rcParams.update({"font.size": 11, "text.color": TEXT})
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.3), dpi=220)
    area = 330.0 / np.abs(np.concatenate([v0, vs])).max()      # marker area (pt^2) per unit |component|
    for ax, v, title in (
            (axes[0], v0, fr"$v_0$: $\lambda_0={lam[0]:.1f}$, $\omega_0={1/np.sqrt(delta*lam[0]):.2f}$"
                          f"\n{100*w0:.1f} % uniform vector"),
            (axes[1], vs, fr"$v_{{{n-1}}}$: $\lambda={lam[-1]:.2f}$, $\omega={1/np.sqrt(delta*lam[-1]):.2f}$"
                          f"\n{100*ws:.0f} % staggered pattern")):
        for a, b in bonds:
            ax.plot(*pos[[a, b]].T, color=GRID, lw=2.0, zorder=1)
        ax.scatter(*pos.T, s=area * np.abs(v), c=[POSITIVE if x > 0 else NEGATIVE for x in v],
                   edgecolors="white", linewidths=1.0, zorder=3)
        ax.scatter(*pos.T, s=area * uniform, facecolors="none", edgecolors=MUTED, linewidths=0.7,
                   linestyles=(0, (2, 2)), zorder=4)
        ax.set_title(title, fontsize=11, color=TEXT)
        ax.set_aspect("equal")
        ax.set_xlim(pos[:, 0].min() - 0.45, pos[:, 0].max() + 0.45)
        ax.set_ylim(pos[:, 1].min() - 0.45, pos[:, 1].max() + 0.45)
        ax.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker="o", ms=9, mfc=POSITIVE, mec="white", label="positive component"),
               Line2D([], [], ls="", marker="o", ms=9, mfc=NEGATIVE, mec="white", label="negative component"),
               Line2D([], [], ls="", marker="o", ms=9, mfc="none", mec=MUTED, label=r"uniform size $1/\sqrt{N_x}$")]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=10)
    fig.text(0.5, 0.155, "marker area ∝ |component|", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    out = os.path.join(IMAGES, "perylene_modes.png")
    fig.savefig(out)
    print(f"{out}: v0 components {v0.min():.3f} ... {v0.max():.3f} (uniform {uniform:.3f}), "
          f"overlap^2 with uniform {w0:.4f}; last mode overlap^2 with staggered {ws:.4f}; "
          f"u^T V u = {np.ones(n) @ ppp_ed.interaction_matrix(yml) @ np.ones(n) / n:.2f}")


if __name__ == "__main__":
    os.makedirs(IMAGES, exist_ok=True)
    convert_pdfs()
    gauge_modes()
    mode_shapes()
