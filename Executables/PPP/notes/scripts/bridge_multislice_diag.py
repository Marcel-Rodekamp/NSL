"""Multi-slice bridge diagnostic on stored configurations of a bridge chain (deck 7, "multi-slice bridge").

    python bridge_multislice_diag.py chain.yml stage1.h5 [first last | --random N --seed S] [-o out.npz]

For every configuration: the generalized densities n_i(t) = 1 - G_ii(t,t) at every time slice, from phi alone
(G = M_p^-1, transfer matrices as in ppp_reduce.logdet_particle), and from them, with y = Im n:
  lambda_b    Lambda at the bridge slice b (checked against config/bridge_lambda of the stage-1 file),
  lambda_all  the same bridge value averaged over all slices, cos^2(theta/2) + 4 sin^2(theta/2) mean_{t,i} y^2,
  sx2         <s_x^2>_phi = mean_{t,i} (1/2 + 2 y^2), the estimator of the notebook's <s_x^2>,
  peak        the slice t with the largest mean_i y_i(t)^2, and that value; lam_t: Lambda_t at every slice.
Re n = 1/2 on every configuration (mu = 0, bipartite) is checked as well.
Run on Jureca for the MN Nt = 288 chain c25 (window 1975-1994, and 200 random configurations, seed 1); the outputs
mn288_c25_{outlier,random}.npz feed bridge_multislice_figures.py --diag.
"""
import argparse
import os
import sys

import h5py
import numpy as np
import scipy.linalg as sla
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import ppp_reduce as pr  # noqa: E402


def qdt_product(bs):
    """b_{k-1} ... b_1 b_0 of the list bs as (Q, D, T) with pivoted QR at every step"""
    n = bs[0].shape[0]
    Q, D, T = np.eye(n, dtype=complex), np.ones(n), np.eye(n, dtype=complex)
    for b in bs:
        Qn, R, piv = sla.qr((b @ Q) * D[None, :], pivoting=True)
        d = np.abs(np.diag(R))
        T, Q, D = (R / d[:, None]) @ T[piv], Qn, d
    return Q, D, T


def inv_one_plus(Q, D, T):
    """(1 + Q D T)^-1 = T^-1 (Db^-1 Q^+ T^-1 + Ds)^-1 Db^-1 Q^+, with D = Db Ds split into large and small parts"""
    Db, Ds = np.maximum(D, 1.0), np.minimum(D, 1.0)
    Tinv = np.linalg.inv(T)
    X = (Q.conj().T @ Tinv) / Db[:, None] + np.diag(Ds)
    return Tinv @ np.linalg.solve(X, (Q.conj().T / Db[:, None]))


def densities(eK, phi, fresh=12):
    """n_i(t) = 1 - G_ii(t,t) for every t: G(t,t) = (1 + b_t b_{t-1} ... b_{t+1})^-1, b_t = e^{delta K} diag(e^{i phi_{t-1}})
    (t = 0 is the product of ppp_reduce.logdet_particle). Computed afresh every `fresh` slices, in between
    G(t+1,t+1) = b_{t+1} G(t,t) b_{t+1}^-1 (exact; the condition number of b is e^{delta (lambda_max - lambda_min)})."""
    Nt, n = phi.shape
    b = [eK * np.exp(1j * phi[(t - 1) % Nt])[None, :] for t in range(Nt)]
    out = np.empty((Nt, n), complex)
    G = None
    for t in range(Nt):
        if t % fresh == 0:
            G = inv_one_plus(*qdt_product([b[(t + 1 + s) % Nt] for s in range(Nt)]))   # rightmost b_{t+1}
        else:
            G = b[t] @ G @ np.linalg.inv(b[t])
        out[t] = 1.0 - np.diagonal(G)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yml")
    ap.add_argument("stage1")
    ap.add_argument("range", nargs="*", type=int, help="first and last configuration id")
    ap.add_argument("--random", type=int, default=0, help="this many random configurations instead of a range")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--h5", help="ensemble file (default: fileIO: h5file of the yml, relative to the working dir)")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    yml = yaml.safe_load(open(args.yml))
    st = pr.static_setup(yml)
    h5in = args.h5 or yml["fileIO"]["h5file"]
    eK = sla.expm(st["delta"] * st["K"])
    with h5py.File(args.stage1, "r") as f:
        lam_stored = f["config/bridge_lambda"][()]
        ids_stored = f["config/id"][()]
        theta, b = float(f.attrs["bridge_theta"]), int(f.attrs["bridge_slice"])
    stored = dict(zip(ids_stored.tolist(), lam_stored.tolist()))
    with h5py.File(h5in, "r") as f:
        mc = f[f"{st['name']}/markovChain"]
        n_cfg = sum(1 for k in mc.keys() if k.isdigit())
        if args.random:
            ids = np.sort(np.random.default_rng(args.seed).choice(n_cfg, args.random, replace=False))
        else:
            ids = np.arange(args.range[0], args.range[1] + 1)
        rows = []
        c2, s2 = np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2
        for i in ids:
            phi = np.real(mc[f"{i}/phi"][()]).reshape(st["Nt"], st["Nx"])
            n = densities(eK, phi)
            y2 = n.imag ** 2                                       # (Nt, Nx)
            lam_t = c2 + 4 * s2 * y2.mean(axis=1)                  # bridge value of every slice
            rows.append(dict(id=int(i), lam_b=lam_t[b], lam_stored=stored.get(int(i), np.nan), lam_all=lam_t.mean(),
                             sx2=np.mean(0.5 + 2 * y2), peak=int(np.argmax(lam_t)), lam_max=lam_t.max(),
                             re_dev=float(np.max(np.abs(n.real - 0.5) / np.maximum(1, np.abs(n)))), lam_t=lam_t))
            r = rows[-1]
            print(f"{r['id']:5d}  Lambda_b {r['lam_b']:10.4g} (stored {r['lam_stored']:10.4g})  Lambda_all {r['lam_all']:9.4g}"
                  f"  max_t Lambda_t {r['lam_max']:9.4g} at t={r['peak']:3d}  <s_x^2>_phi {r['sx2']:9.4g}"
                  f"  sx2/Lambda_b {r['sx2'] / r['lam_b']:9.4g}  sx2/Lambda_all {r['sx2'] / r['lam_all']:6.3f}"
                  f"  |Re n - 1/2| {r['re_dev']:.1e}", flush=True)
    dev = max(abs(r["lam_b"] - r["lam_stored"]) / r["lam_stored"] for r in rows if np.isfinite(r["lam_stored"]))
    print(f"max relative deviation of Lambda_b from the stored bridge_lambda: {dev:.2e}")
    if args.output:
        np.savez(args.output, **{k: np.array([r[k] for r in rows]) for k in rows[0]})


if __name__ == "__main__":
    main()
