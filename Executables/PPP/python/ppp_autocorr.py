#!/usr/bin/env python3
"""Integrated autocorrelation times of a PPP ensemble (Gamma method, U. Wolff,
Comput. Phys. Commun. 156 (2004) 143).

    python ppp_autocorr.py run.yml [run.h5] [--discard N] [--seconds-per-trajectory s] [--json out.json]

Observables, as functions of the Markov time (one stored configuration = one step):

  S          real part of the total action (`actVal`)
  S_gauge    1/2 sum_t phi_t^T (delta V)^{-1} phi_t
  S_fermion  S - S_gauge
  a_k, a_k^2 amplitude of the time-summed field along the k-th eigenvector v_k of V,
             a_k = sum_t phi_t . v_k / sqrt(Nt delta lambda_k)   (k = 0: largest lambda),
             normalized to unit variance for the gauge action alone
  G(tau)     if position-space propagators were measured (Executables/Hubbard/propagator,
             measurements: type: spatial or both): the site-averaged
             particle propagator Re 1/Nx sum_x G_xx(tau), averaged over time sources

The integrated autocorrelation time is also estimated from bin averages (bins of N/40, N/20,
N/10): rare long excursions of the chain show up there but not in the automatically
windowed Gamma-method estimate. With --seconds-per-trajectory the cost of one independent
sample, 2 tau * s with the larger of the estimates, is printed.
"""

import argparse
import json
import os
import sys

import numpy as np
import h5py
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppp_ed  # noqa: E402
import ppp_compare as pc  # noqa: E402


def gamma_method(a, S=1.5):
    """tau_int, its error and the summation window (Wolff's automatic windowing)"""
    a = np.asarray(a, dtype=float)
    N = len(a)
    d = a - a.mean()
    if N < 10 or np.allclose(d, 0):
        return 0.5, 0.0, 0
    # autocorrelation function via FFT
    f = np.fft.rfft(d, n=2 * N)
    gamma = np.fft.irfft(f * np.conj(f))[:N] / (N - np.arange(N))
    rho = gamma / gamma[0]
    tau = 0.5
    W = 0
    for W in range(1, N // 2):
        tau = 0.5 + rho[1:W + 1].sum()
        if tau <= 0.5:
            tau = 0.5
            break
        tauW = S / np.log((2 * tau + 1) / (2 * tau - 1))
        g = np.exp(-W / tauW) - tauW / np.sqrt(W * N)
        if g < 0:
            break
    dtau = tau * np.sqrt((4 * W + 2) / N)
    return tau, dtau, W


def tau_binned(a, binsize):
    """tau_int from the variance of bin averages, 1/2 b var(bin means)/var(a)"""
    a = np.asarray(a, dtype=float)
    nb = len(a) // binsize
    if nb < 2 or a.var() == 0:
        return np.nan
    means = a[: nb * binsize].reshape(nb, binsize).mean(axis=1)
    return 0.5 * binsize * means.var(ddof=1) / a.var(ddof=1)


def observables(yml, h5file, discard=0, chain="markovChain"):
    name = yml["system"]["name"]
    Nt = int(yml["system"]["Nt"])
    beta = float(yml["system"]["beta"])
    delta = beta / Nt
    V = ppp_ed.interaction_matrix(yml)
    lam, Q = np.linalg.eigh(V)
    order = np.argsort(lam)[::-1]          # k = 0: largest eigenvalue
    lam, Q = lam[order], Q[:, order]
    Ainv = np.linalg.inv(delta * V)

    obs = {}
    with h5py.File(h5file, "r") as f:
        g = f[f"{name}/{chain}"]
        ids = pc.config_ids(g)[discard:]
        phi = np.array([pc.read_tensor(g[f"{i}/phi"]).real for i in ids])     # (Ncfg, Nt, Nx)
        act = np.array([np.asarray(g[f"{i}/actVal"][()]).real for i in ids])
        obs["S"] = act
        obs["S_gauge"] = 0.5 * np.einsum("ntx,xy,nty->n", phi, Ainv, phi)
        obs["S_fermion"] = obs["S"] - obs["S_gauge"]
        amp = phi.sum(axis=1) @ Q / np.sqrt(Nt * delta * lam)                # (Ncfg, Nx)
        for k in range(len(lam)):
            obs[f"a_{k}"] = amp[:, k]
            obs[f"a_{k}^2"] = amp[:, k] ** 2

        measured = [i for i in ids if pc.has_spatial(g[f"{i}"])]
        if len(measured) == len(ids) and measured:
            G = np.array([pc.averaged_column(pc.source_columns(g[f"{i}"], "particle"), Nt) for i in ids])
            Gloc = np.einsum("ntxx->nt", G).real / G.shape[-1]
            for tau in sorted({1, 2, 4, Nt // 8, Nt // 4, Nt // 2}):
                obs[f"G({tau})"] = Gloc[:, tau]
        elif measured:
            print(f"# propagators measured on {len(measured)} of {len(ids)} configurations: skipped")
    return obs, lam, len(ids)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("yml")
    p.add_argument("h5", nargs="?", help="default: fileIO: h5file of the yml file")
    p.add_argument("--discard", type=int, default=0, help="number of leading configurations to drop")
    p.add_argument("--chain", default="markovChain")
    p.add_argument("--seconds-per-trajectory", type=float, default=None)
    p.add_argument("--json", default=None, help="write the results to this file")
    args = p.parse_args(argv)

    with open(args.yml) as fh:
        yml = yaml.safe_load(fh)
    h5file = args.h5 or yml["fileIO"]["h5file"]
    if not args.h5 and not os.path.exists(h5file):
        # relative to the yml file
        h5file = os.path.join(os.path.dirname(os.path.abspath(args.yml)), h5file)
    obs, lam, N = observables(yml, h5file, args.discard, args.chain)

    bins = [b for b in (N // 40, N // 20, N // 10) if b >= 20]
    print(f"# {h5file}: {N} configurations, lambda(V) = {lam[-1]:.3f} ... {lam[0]:.3f}")
    print("# tau_int: Gamma method; tau_bin(b): from bins of b configurations, which also sees slow")
    print("# tails the automatic window misses ('!' if tau_bin > 2 tau_int + 1: use the larger value)")
    head = f"# {'observable':>10s} {'tau_int':>9s} {'err':>7s} {'W':>5s} " + " ".join(f"{'bin '+str(b):>8s}" for b in bins)
    if args.seconds_per_trajectory:
        head += f" {'s/indep.':>9s}"
    print(head)
    res = {}
    for key, a in obs.items():
        tau, dtau, W = gamma_method(a)
        tb = [tau_binned(a, b) for b in bins]
        slow = any(t > 2 * tau + 1 for t in tb if np.isfinite(t))
        res[key] = {"tau": tau, "dtau": dtau, "W": W, "tau_bin": dict(zip(map(str, bins), tb))}
        line = f"  {key:>10s} {tau:9.2f} {dtau:7.2f} {W:5d} " + " ".join(f"{t:8.2f}" for t in tb) + (" !" if slow else "  ")
        if args.seconds_per_trajectory:
            line += f" {2 * max([tau] + [t for t in tb if np.isfinite(t)]) * args.seconds_per_trajectory:9.3f}"
        print(line)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"N": N, "lambda": lam.tolist(), "seconds_per_trajectory": args.seconds_per_trajectory,
                       "tau": res}, fh, indent=1)


if __name__ == "__main__":
    main()
