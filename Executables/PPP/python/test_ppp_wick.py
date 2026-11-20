#!/usr/bin/env python3
"""Checks ppp_wick.py against exact Fock-space traces for one fixed auxiliary field.

    python test_ppp_wick.py            # prints the deviations, exit code 1 on failure

For a fixed field the transfer matrices of NSL's HubbardExp fermion matrix (ppp_ed.fermion_matrix)
are exponentials of fermion bilinears, so every correlator can be computed exactly in the Fock
space of a small lattice (4-site chain, both spins, 256 states) and compared with the Wick
contractions of G = M^{-1}.
"""

import os
import sys

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppp_ed  # noqa: E402
import ppp_wick as wick  # noqa: E402

TOL = 1e-11


def build(n=4, Nt=6, delta=0.4, seed=3):
    rng = np.random.default_rng(seed)
    K = np.zeros((n, n))
    for i in range(n - 1):
        K[i, i + 1] = K[i + 1, i] = 1.0
    phi = rng.normal(0.0, 0.8, (Nt, n))
    m, dim = 2 * n, 2 ** (2 * n)
    sm, Z2, I2 = np.array([[0, 1], [0, 0]]), np.diag([1, -1]), np.eye(2)

    def ann(j):
        ops = [Z2] * j + [sm] + [I2] * (m - j - 1)
        out = ops[0]
        for o in ops[1:]:
            out = np.kron(out, o)
        return out.astype(complex)

    c = [ann(j) for j in range(m)]
    cd = [x.conj().T for x in c]
    mode = {"u": 0, "d": n}

    def bil(A, s):
        return sum(A[x, y] * cd[x + mode[s]] @ c[y + mode[s]] for x in range(n) for y in range(n) if A[x, y] != 0)

    def gauss(A):
        return sla.expm(bil(A, "u") + bil(A, "d"))

    eK = gauss(delta * K)
    T = [eK @ gauss(np.diag(1j * phi[(t - 1) % Nt])) for t in range(Nt)]   # b_t = e^{delta K} e^{i phi_{t-1}}
    prefix = [np.eye(dim, dtype=complex)]                                   # prefix[t] = T_t ... T_1
    for t in range(1, Nt + 1):
        prefix.append(T[t % Nt] @ prefix[-1])
    Z = np.trace(prefix[Nt])

    def U(t2, t1):
        return prefix[t2] @ np.linalg.inv(prefix[t1])

    def corr(A_late, t, B_early, t0):        # Tr[U(Nt<-t) A U(t<-t0) B U(t0<-0)] / Z, t >= t0
        return np.trace(U(Nt, t) @ A_late @ U(t, t0) @ B_early @ prefix[t0]) / Z

    def op(u, s, dag):                       # c^+_{u,s} = sum_x u_x c^+_{x,s}, or c_{u,s}
        return sum(u[x] * (cd if dag else c)[x + mode[s]] for x in range(n))

    def exc(A, s1, s2):                      # sum_xy A_xy c^+_{x,s1} c_{y,s2}
        return sum(A[x, y] * cd[x + mode[s1]] @ c[y + mode[s2]] for x in range(n) for y in range(n) if A[x, y] != 0)

    G = np.linalg.inv(ppp_ed.fermion_matrix(K, phi, delta, "particle")).reshape(Nt, n, Nt, n)
    Gh = np.linalg.inv(ppp_ed.fermion_matrix(K, phi, delta, "hole")).reshape(Nt, n, Nt, n)
    return dict(n=n, Nt=Nt, K=K, c=c, cd=cd, corr=corr, op=op, exc=exc, G=G, Gh=Gh, dim=dim)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(1e-300, np.max(np.abs(b))))


def main():
    L = build()
    n, Nt, G, Gh, corr, c, cd = L["n"], L["Nt"], L["G"], L["Gh"], L["corr"], L["c"], L["cd"]
    I = np.eye(L["dim"])
    results = {}

    # one-body conventions
    results["G(t,t0) = <c(t) c+(t0)>, t > t0"] = max(
        rel(G[t, :, t0, :], [[corr(c[x], t, cd[y], t0) for y in range(n)] for x in range(n)])
        for t in range(Nt) for t0 in range(t))
    results["G(t,t) = <c c+>"] = max(
        rel(G[t, :, t, :], [[corr(c[x] @ cd[y], t, I, t) for y in range(n)] for x in range(n)]) for t in range(Nt))
    results["hole: G_h(t0,t)^T = -G(t,t0)"] = max(rel(Gh[t0, :, t, :].T, -G[t, :, t0, :]) for t in range(Nt) for t0 in range(t))
    results["hole: G_h(t,t)^T = 1 - G(t,t)"] = max(rel(Gh[t, :, t, :].T, np.eye(n) - G[t, :, t, :]) for t in range(Nt))

    # equal-time spin and charge correlations
    nu = lambda x: cd[x] @ c[x]                                       # noqa: E731
    nd = lambda x: cd[x + n] @ c[x + n]                               # noqa: E731
    dev_ss = dev_qq = 0.0
    for t in range(Nt):
        _, ss, qq = wick.equal_time(G[t, :, t, :])
        ss_f = [[corr((nu(x) - nd(x)) @ (nu(y) - nd(y)), t, I, t) for y in range(n)] for x in range(n)]
        qq_f = [[corr((nu(x) + nd(x) - I) @ (nu(y) + nd(y) - I), t, I, t) for y in range(n)] for x in range(n)]
        dev_ss, dev_qq = max(dev_ss, rel(ss, ss_f)), max(dev_qq, rel(qq, qq_f))
    results["equal time <s_x s_y>"] = dev_ss
    results["equal time <q_x q_y>"] = dev_qq

    # orbitals: conduction (eps = -lambda > 0) and valence partners u_{-p} = (-1)^x u_p
    lam, W = np.linalg.eigh(L["K"])
    sub = (-1.0) ** np.arange(n)
    cond = [W[:, k] for k in range(n) if lam[k] < 0]
    val = [sub * u for u in cond]
    exc = L["exc"]
    ops = {"pair (0,0)": np.outer(cond[0], val[0]),
           "pair (0,1)": np.outer(cond[0], val[1]),
           "I=0-like (0,1)+(1,0)": np.outer(cond[0], val[1]) + np.outer(cond[1], val[0]),
           "I=1-like (0,1)-(1,0)": np.outer(cond[0], val[1]) - np.outer(cond[1], val[0])}
    t0 = 1
    dev = {"exciton triplet S_z=+1": 0.0, "exciton triplet S_z=0": 0.0, "exciton singlet": 0.0}
    for ki, Ai in ops.items():
        for kj, Aj in ops.items():
            for t in range(t0 + 1, Nt):
                X, S = wick.exciton(Ai, Aj, G[t, :, t0, :], G[t0, :, t, :], G[t, :, t, :], G[t0, :, t0, :])
                Tp = corr(exc(Ai, "u", "d").conj().T, t, exc(Aj, "u", "d"), t0)
                T0 = corr((exc(Ai, "u", "u") - exc(Ai, "d", "d")).conj().T / np.sqrt(2), t,
                          (exc(Aj, "u", "u") - exc(Aj, "d", "d")) / np.sqrt(2), t0)
                Sf = corr((exc(Ai, "u", "u") + exc(Ai, "d", "d")).conj().T / np.sqrt(2), t,
                          (exc(Aj, "u", "u") + exc(Aj, "d", "d")) / np.sqrt(2), t0)
                dev["exciton triplet S_z=+1"] = max(dev["exciton triplet S_z=+1"], rel(X, Tp))
                dev["exciton triplet S_z=0"] = max(dev["exciton triplet S_z=0"], rel(X, T0))
                dev["exciton singlet"] = max(dev["exciton singlet"], rel(S, Sf))
    results.update(dev)

    # pp and hh: B for c^+_{x,up} c^+_{y,down} (resp. c_{x,up} c_{y,down})
    def pp_op(B):
        return sum(B[x, y] * cd[x] @ cd[y + n] for x in range(n) for y in range(n) if B[x, y] != 0)

    def hh_op(B):
        return sum(B[x, y] * c[x] @ c[y + n] for x in range(n) for y in range(n) if B[x, y] != 0)

    Bs = {"singlet (0,1)": wick.pair_matrix(cond[0], cond[1], +1), "triplet (0,1)": wick.pair_matrix(cond[0], cond[1], -1),
          "same orbital (0,0)": wick.pair_matrix(cond[0], cond[0])}
    Bv = {"singlet (0,1)": wick.pair_matrix(val[0], val[1], +1), "triplet (0,1)": wick.pair_matrix(val[0], val[1], -1),
          "same orbital (0,0)": wick.pair_matrix(val[0], val[0])}
    # deviations relative to the largest element of the channel: singlet-triplet cross elements
    # vanish on every configuration (spin symmetry) and are compared on that scale
    pw, pf, hw, hf = [], [], [], []
    for t in range(t0 + 1, Nt):
        for B in Bs.values():
            for B2 in Bs.values():
                pw.append(wick.pp(B, B2, G[t, :, t0, :]))
                pf.append(corr(pp_op(B).conj().T, t, pp_op(B2), t0))
        for B in Bv.values():
            for B2 in Bv.values():
                hw.append(wick.hh(B, B2, G[t0, :, t, :]))
                hf.append(corr(hh_op(B).conj().T, t, hh_op(B2), t0))
    results["pp (singlet, triplet, same orbital; all sink/source pairs)"] = rel(pw, pf)
    results["hh (singlet, triplet, same orbital; all sink/source pairs)"] = rel(hw, hf)

    # equal times, tau = 0 (every time slice): <O O^+> in operator order, sink left of source, from
    # Gf = G(t,t) and the backward propagator G(t,t) - 1 (backward_equal_time: the contact terms)
    dev0 = {k: 0.0 for k in ("triplet S_z=+1", "triplet S_z=0", "singlet")}
    pw0, pf0, hw0, hf0 = [], [], [], []
    for tt in range(Nt):
        Gt = G[tt, :, tt, :]
        Gb0 = wick.backward_equal_time(Gt)
        for Ai in ops.values():
            for Aj in ops.values():
                X, S = wick.exciton(Ai, Aj, Gt, Gb0, Gt, Gt)
                Tp = corr(exc(Ai, "u", "d").conj().T, tt, exc(Aj, "u", "d"), tt)
                T0 = corr((exc(Ai, "u", "u") - exc(Ai, "d", "d")).conj().T / np.sqrt(2), tt,
                          (exc(Aj, "u", "u") - exc(Aj, "d", "d")) / np.sqrt(2), tt)
                Sf = corr((exc(Ai, "u", "u") + exc(Ai, "d", "d")).conj().T / np.sqrt(2), tt,
                          (exc(Aj, "u", "u") + exc(Aj, "d", "d")) / np.sqrt(2), tt)
                for k, w, f in (("triplet S_z=+1", X, Tp), ("triplet S_z=0", X, T0), ("singlet", S, Sf)):
                    dev0[k] = max(dev0[k], rel(w, f))
        for B in Bs.values():
            for B2 in Bs.values():
                pw0.append(wick.pp(B, B2, Gt))
                pf0.append(corr(pp_op(B).conj().T, tt, pp_op(B2), tt))
        for B in Bv.values():
            for B2 in Bv.values():
                hw0.append(wick.hh(B, B2, Gb0))
                hf0.append(corr(hh_op(B).conj().T, tt, hh_op(B2), tt))
    results.update({f"tau = 0: exciton {k}": v for k, v in dev0.items()})
    results["tau = 0: pp (forward only, no contact term)"] = rel(pw0, pf0)
    results["tau = 0: hh (two backward propagators)"] = rel(hw0, hf0)

    # vectorized call over t (leading axis) agrees with single calls; G[t0, :, Ts, :] has shape (len(Ts), n, n)
    Ts = np.arange(t0 + 1, Nt)
    Xv, Sv = wick.exciton(ops["pair (0,1)"], ops["pair (0,1)"], G[Ts, :, t0, :], G[t0, :, Ts, :],
                          G[Ts, :, Ts, :], G[t0, :, t0, :])
    Xs = [wick.exciton(ops["pair (0,1)"], ops["pair (0,1)"], G[t, :, t0, :], G[t0, :, t, :], G[t, :, t, :], G[t0, :, t0, :])[1]
          for t in Ts]
    results["vectorized over t"] = rel(Sv, Xs)

    # pair-basis contractions (exciton_pairs, exciton_expectations, pp_pairs) against the general functions
    # (checked against Fock space above), on the 4-site chain with all conduction x valence pairs
    U4 = np.array(cond + val).T                                   # orbital basis: conduction, then valence
    ci, vi = np.repeat([0, 1], 2), np.tile([2, 3], 2)             # pairs (c, v)
    Ts = np.arange(t0 + 1, Nt)
    Fo = U4.T @ G[Ts, :, t0, :] @ U4
    Bo = U4.T @ G[t0, :, Ts, :] @ U4
    Eo_t = U4.T @ wick.density_matrix(G[Ts, :, Ts, :]) @ U4
    Eo_0 = U4.T @ wick.density_matrix(G[t0, :, t0, :]) @ U4
    Xp = wick.exciton_pairs(Fo, Bo, ci, vi)
    o_t, _ = wick.exciton_expectations(Eo_t, ci, vi)
    _, ob_0 = wick.exciton_expectations(Eo_0, ci, vi)
    Sp = Xp + 2 * o_t[:, :, None] * ob_0[None, None, :]
    Xg = np.zeros_like(Xp)
    Sg = np.zeros_like(Xp)
    for i in range(len(ci)):
        for j in range(len(ci)):
            Ai, Aj = np.outer(U4[:, ci[i]], U4[:, vi[i]]), np.outer(U4[:, ci[j]], U4[:, vi[j]])
            Xg[:, i, j], Sg[:, i, j] = wick.exciton(Ai, Aj, G[Ts, :, t0, :], G[t0, :, Ts, :], G[Ts, :, Ts, :], G[t0, :, t0, :])
    results["exciton_pairs (triplet) = exciton"] = rel(Xp, Xg)
    results["exciton_pairs + expectations (singlet) = exciton"] = rel(Sp, Sg)
    # tau = 0 as in stage 2: Bo = U^T G(t0,t0) U - 1 in the orbital basis, against Fock space directly
    Fo0 = U4.T @ G[t0, :, t0, :] @ U4
    Xp0 = wick.exciton_pairs(Fo0, Fo0 - np.eye(n), ci, vi)
    Xf0 = np.array([[corr(exc(np.outer(U4[:, ci[i]], U4[:, vi[i]]), "u", "d").conj().T, t0,
                          exc(np.outer(U4[:, ci[j]], U4[:, vi[j]]), "u", "d"), t0) for j in range(len(ci))]
                    for i in range(len(ci))])
    results["tau = 0: exciton_pairs with Bo = U^T G U - 1 (stage 2)"] = rel(Xp0, Xf0)
    for sgn, name in ((+1, "singlet"), (-1, "triplet")):
        a = np.array([0, 0, 1] if sgn > 0 else [0])
        b = np.array([0, 1, 1] if sgn > 0 else [1])
        Pp = wick.pp_pairs(Fo, a, b, sgn)
        Pg = np.array([[wick.pp(wick.pair_matrix(U4[:, a[i]], U4[:, b[i]], sgn), wick.pair_matrix(U4[:, a[j]], U4[:, b[j]], sgn),
                                G[Ts, :, t0, :]) for j in range(len(a))] for i in range(len(a))]).transpose(2, 0, 1)
        results[f"pp_pairs ({name}) = pp"] = rel(Pp, Pg)

    # disconnected exciton part of stage 2: circular cross-correlation over all time slices (complex data)
    import ppp_reduce2  # noqa: E402
    o = rng_cc = np.random.default_rng(11)
    o = rng_cc.normal(size=(16, 3)) + 1j * rng_cc.normal(size=(16, 3))
    ob = rng_cc.normal(size=(16, 3)) + 1j * rng_cc.normal(size=(16, 3))
    direct = np.array([sum(np.outer(o[(t0 + t) % 16], ob[t0]) for t0 in range(16)) / 16 for t in range(16)])
    results["stage 2 cross-correlation (FFT) = direct sum"] = rel(ppp_reduce2.cross_correlation(o, ob), direct)

    # fermion determinant from the stabilized transfer-matrix product (ppp_reduce.logdet_particle)
    # against the dense determinant, on a 20-site ring at beta = 32 (large condition numbers)
    import ppp_reduce  # noqa: E402
    rng = np.random.default_rng(7)
    nr, Ntr, dr = 20, 128, 0.25
    Kr = np.zeros((nr, nr))
    for i in range(nr):
        Kr[i, (i + 1) % nr] = Kr[(i + 1) % nr, i] = 1.0
    dev_l = dev_p = 0.0
    for scale in (0.0, 0.5, 1.5):
        phr = rng.normal(0.0, scale, (Ntr, nr)) if scale else np.zeros((Ntr, nr))
        with np.errstate(all="ignore"):
            s, l = np.linalg.slogdet(ppp_ed.fermion_matrix(Kr, phr, dr, "particle"))
        ld = ppp_reduce.logdet_particle(Kr, phr, dr)
        dev_l = max(dev_l, abs(ld.real - l) / abs(l))
        dev_p = max(dev_p, abs(np.angle(np.exp(1j * ld.imag) / s)))
    results["log|det M_p|, transfer matrices vs dense"] = dev_l
    results["phase of det M_p, transfer matrices vs dense"] = dev_p

    failed = False
    for k, v in results.items():
        ok = v < TOL
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {k:60s} {v:.1e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
