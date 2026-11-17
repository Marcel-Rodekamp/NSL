#!/usr/bin/env python3
r"""Exact diagonalization benchmarks for NSL's PPP simulations.

Model (energies in units of t, see src/NSL/Action/Implementations/PPP_Interaction.tpp)

    H = -sum_{xy,s} K_xy c^+_{xs} c_{ys} + 1/2 sum_xy V_xy q_x q_y ,  q_x = n_xu + n_xd - 1

NSL samples the exponential (charge basis) discretization with the PPP gauge
action. Integrating out the auxiliary field slice by slice is exact, so the
ensemble averages of NSL's fermion-matrix inverses are the propagators of the
transfer matrix

    W = exp(delta K^) exp(-delta H_V) ,   delta = beta/Nt ,

    <M_p^{-1}[t,0]>_xy = Tr[ W^(Nt-t) c_{x,up}   W^t c^+_{y,up} ] / Tr[ W^Nt ]   (particles)
    <M_h^{-1}[t,0]>_xy = Tr[ W^(Nt-t) c^+_{x,dn} W^t c_{y,dn}   ] / Tr[ W^Nt ]   (holes)

for t = 0..Nt-1 (`self_test` checks this mapping against an explicit fermion
matrix). These lattice propagators have no Trotter error relative to the HMC
data. The continuum propagators (exp(-tau H)) are also available.

Two backends:
  dense     full diagonalization of every (N_up, N_dn) sector; exact. Benzene.
  lowlying  Lanczos for the dominant transfer-matrix eigenstates of each sector,
            exact propagation in between. Sectors are skipped only if a rigorous
            upper bound of their weight (lambda/lambda_0)^(Nt/2) is below `tol`;
            eigenstates are dropped by the same criterion. Check convergence by
            varying --tol (naphthalene, beta=20: 1e-10 -> 1e-13 changes G by
            1e-11). Naphthalene.

Usage
    python ppp_ed.py naphthalene.yml -o naphthalene_ed.h5 [--backend lowlying]
    python ppp_ed.py --self-test
"""

import argparse
import itertools
import math
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

E2_EV_ANGSTROM = 14.399645

# =============================================================================
# Model input (mirrors NSL::Lattice::Generic and NSL::PPP)
# =============================================================================


def hopping_matrix(sysyml):
    n = int(sysyml["nions"])
    K = np.zeros((n, n))
    hop = sysyml["hopping"]
    adj = sysyml["adjacency"]
    for i, (a, b) in enumerate(adj):
        h = float(hop[i]) if isinstance(hop, (list, tuple)) else float(hop)
        K[a, b] = K[b, a] = h
    return K


def interaction_matrix(yml):
    sysyml, ppp = yml["system"], yml["PPP"]
    n = int(sysyml["nions"])
    U = float(sysyml["U"])
    form = "".join(c for c in str(ppp["form"]).lower() if c not in "-_ ")
    if form in ("explicit", "matrix"):
        return np.array(ppp["V"], dtype=float)
    t_eV = float(ppp.get("hopping energy", 2.7))
    length = float(ppp.get("length unit", 1.4))
    eps = float(ppp.get("dielectric", 1.0))
    e2 = float(ppp.get("e2", E2_EV_ANGSTROM))
    pos = np.array(sysyml["positions"], dtype=float)
    r = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    x = U * t_eV * eps * length * r / e2
    if form == "ohno":
        V = U / np.sqrt(1 + x**2)
    elif form in ("mataganishimoto", "mn"):
        V = U / (1 + x)
    elif form in ("onsite", "hubbard"):
        V = np.zeros((n, n))
    else:
        raise ValueError(f"unknown PPP form {ppp['form']}")
    np.fill_diagonal(V, U)
    return V


# =============================================================================
# Fock space of one spin species, organised by particle number
# =============================================================================


class SpinSpace:
    """States c^+_{i1} ... c^+_{ik}|0> with i1 < ... < ik, stored as bit patterns."""

    def __init__(self, n):
        self.n = n
        allstates = np.arange(2**n, dtype=np.int64)
        pop = np.array([bin(s).count("1") for s in allstates])
        self.states = {k: allstates[pop == k] for k in range(n + 1)}
        self.index = {k: {int(s): i for i, s in enumerate(self.states[k])} for k in range(n + 1)}
        self.occ = {k: ((self.states[k][:, None] >> np.arange(n)) & 1).astype(float) for k in range(n + 1)}

    def dim(self, k):
        return len(self.states[k]) if 0 <= k <= self.n else 0

    def annihilator(self, x, k):
        """c_x as a (dim(k-1), dim(k)) matrix."""
        C = np.zeros((self.dim(k - 1), self.dim(k)))
        if k == 0:
            return C
        for j, s in enumerate(self.states[k]):
            s = int(s)
            if (s >> x) & 1:
                sign = -1.0 if bin(s & ((1 << x) - 1)).count("1") % 2 else 1.0
                C[self.index[k - 1][s ^ (1 << x)], j] = sign
        return C

    def one_body(self, A, k):
        """sum_xy A_xy c^+_x c_y in the k-particle sector."""
        d = self.dim(k)
        M = np.zeros((d, d), dtype=np.result_type(A, float))
        for j, s in enumerate(self.states[k]):
            s = int(s)
            for y in range(self.n):
                if not (s >> y) & 1:
                    continue
                s1 = s ^ (1 << y)
                sy = -1 if bin(s & ((1 << y) - 1)).count("1") % 2 else 1
                for x in range(self.n):
                    if A[x, y] == 0 or (s1 >> x) & 1:
                        continue
                    sx = -1 if bin(s1 & ((1 << x) - 1)).count("1") % 2 else 1
                    M[self.index[k][s1 | (1 << x)], j] += A[x, y] * sx * sy
        return M


# =============================================================================
# The PPP transfer matrix in (N_up, N_dn) sectors
# =============================================================================


class PPP:
    """Sector structure: state |I_up, J_dn> = c^+_{I,up} c^+_{J,dn} |0>, row-major in (I, J)."""

    def __init__(self, K, V, beta, Nt):
        self.K, self.V = np.asarray(K, float), np.asarray(V, float)
        self.n = len(K)
        self.beta, self.Nt = float(beta), int(Nt)
        self.delta = self.beta / self.Nt
        self.space = SpinSpace(self.n)
        n = self.n
        self.Ksec = {k: self.space.one_body(self.K, k) for k in range(n + 1)}
        self.Esec = {k: sla.expm(self.delta * self.Ksec[k]) for k in range(n + 1)}
        self.C = {(x, k): self.space.annihilator(x, k) for x in range(n) for k in range(1, n + 1)}
        self.kvals = np.linalg.eigvalsh(self.K)   # ascending

    # --- sector quantities -------------------------------------------------
    def sectors(self):
        return [(a, b) for a in range(self.n + 1) for b in range(self.n + 1)]

    def HV(self, a, b):
        """diagonal of H_V in sector (a,b) as a (dim_a, dim_b) array"""
        q = self.space.occ[a][:, None, :] + self.space.occ[b][None, :, :] - 1.0
        return 0.5 * np.einsum("abx,xy,aby->ab", q, self.V, q)

    def log_lambda_bound(self, a, b):
        """Upper bound of log(largest eigenvalue of W) in sector (a,b)."""
        kmax = self.kvals[::-1]
        bound = self.delta * (kmax[:a].sum() + kmax[:b].sum())
        return bound - self.delta * self.HV(a, b).min()

    def energy_bound(self, a, b):
        """Lower bound of the smallest eigenvalue of H in sector (a,b)."""
        kmax = self.kvals[::-1]
        return -(kmax[:a].sum() + kmax[:b].sum()) + self.HV(a, b).min()

    # --- operators in Kronecker form ----------------------------------------
    def apply_W(self, a, b, v, Dv):
        """W v = (E_a x E_b) (D v);  v has shape (..., dim_a, dim_b)"""
        return self.Esec[a] @ (Dv * v) @ self.Esec[b].T

    # --- dense diagonalization of a sector ----------------------------------
    def dense_sector(self, a, b, continuum):
        hv = self.HV(a, b).ravel()
        if continuum:
            H = -(np.kron(self.Ksec[a], np.eye(self.space.dim(b))) + np.kron(np.eye(self.space.dim(a)), self.Ksec[b]))
            H[np.diag_indices_from(H)] += hv
            E, Q = np.linalg.eigh(H)
            return dict(loglam=-self.delta * E, Q=Q, Dh=np.ones_like(hv))
        Dh = np.exp(-0.5 * self.delta * hv)
        S = Dh[:, None] * np.kron(self.Esec[a], self.Esec[b]) * Dh[None, :]
        lam, Q = np.linalg.eigh(0.5 * (S + S.T))
        if lam.min() <= 0:
            raise RuntimeError("transfer matrix is not positive")
        return dict(loglam=np.log(lam), Q=Q, Dh=Dh)


# =============================================================================
# Propagators
# =============================================================================


def _ops(model, species, x, s, sp):
    """(X_x: sp -> s, Y_x: s -> sp) for the given species as dense matrices."""
    (a, b), (ap, bp) = s, sp
    if species == "particle":      # X = c_{x,up}, Y = c^+_{x,up}, sp = (a+1, b)
        C = model.C[(x, ap)]                          # (dim a, dim a+1)
        X = np.kron(C, np.eye(model.space.dim(b)))
        Y = np.kron(C.T, np.eye(model.space.dim(b)))
    else:                          # X = c^+_{x,dn}, Y = c_{x,dn}, sp = (a, b-1); (-1)^a signs cancel
        C = model.C[(x, b)]                           # (dim b-1, dim b)
        X = np.kron(np.eye(model.space.dim(a)), C.T)
        Y = np.kron(np.eye(model.space.dim(a)), C)
    return X, Y


def _partner(species, s):
    a, b = s
    return (a + 1, b) if species == "particle" else (a, b - 1)


def propagators_dense(model, continuum=False):
    """Exact G_p[t,x,y], G_h[t,x,y] (t = 0..Nt-1) from full diagonalization."""
    n, Nt = model.n, model.Nt
    sec = {s: model.dense_sector(*s, continuum) for s in model.sectors()}
    ll0 = max(d["loglam"].max() for d in sec.values())
    Z = sum(np.exp(Nt * (d["loglam"] - ll0)).sum() for d in sec.values())
    ts = np.arange(Nt)
    out = {}
    for species in ("particle", "hole"):
        G = np.zeros((Nt, n, n))
        for s in model.sectors():
            sp = _partner(species, s)
            if not (0 <= sp[0] <= n and 0 <= sp[1] <= n):
                continue
            A_s, A_p = sec[s], sec[sp]
            # A_x = Q_s^T D_s^{1/2} X_x D_sp^{-1/2} Q_sp ;  B_y = Q_sp^T D_sp^{1/2} Y_y D_s^{-1/2} Q_s
            A, B = [], []
            for x in range(n):
                X, Y = _ops(model, species, x, s, sp)
                A.append(A_s["Q"].T @ (A_s["Dh"][:, None] * X / A_p["Dh"][None, :]) @ A_p["Q"])
                B.append(A_p["Q"].T @ (A_p["Dh"][:, None] * Y / A_s["Dh"][None, :]) @ A_s["Q"])
            A, B = np.array(A), np.array(B)
            L = np.exp((Nt - ts)[:, None] * (A_s["loglam"][None, :] - ll0))    # (t, k)
            M = np.exp(ts[:, None] * (A_p["loglam"][None, :] - ll0))            # (t, l)
            P = np.einsum("xkl,ylk->xykl", A, B)
            G += np.einsum("xykl,tk,tl->txy", P, L, M)
        out[species] = G / Z
    return out


def _lowlying_sector(model, a, b, nev, continuum=False):
    """Largest `nev` eigenpairs of the symmetrized transfer matrix in sector (a,b)."""
    da, db = model.space.dim(a), model.space.dim(b)
    hv = model.HV(a, b)
    Dh = np.exp(-0.5 * model.delta * hv)
    Ea, Eb = model.Esec[a], model.Esec[b]

    def matvec(v):
        v = v.reshape(da, db)
        return (Dh * (Ea @ (Dh * v) @ Eb.T)).ravel()

    d = da * db
    if d <= max(400, 2 * nev + 1):
        S = Dh.ravel()[:, None] * np.kron(Ea, Eb) * Dh.ravel()[None, :]
        lam, Q = np.linalg.eigh(0.5 * (S + S.T))
        lam, Q = lam[::-1][:nev], Q[:, ::-1][:, :nev]
    else:
        op = spla.LinearOperator((d, d), matvec=matvec, dtype=float)
        lam, Q = spla.eigsh(op, k=min(nev, d - 1), which="LA", tol=1e-13)
        order = np.argsort(lam)[::-1]
        lam, Q = lam[order], Q[:, order]
    return dict(loglam=np.log(lam), Q=Q, Dh=Dh, full=(len(lam) == d))


def propagators_lowlying(model, tol=1e-10, nev0=16, verbose=True, chunk=16):
    """Lattice propagators via dominant transfer-matrix eigenstates of each sector.

    For t <= Nt/2 the trace is expanded in eigenstates of the sector of W^(Nt-t),
    for t > Nt/2 (cyclically) in those of W^t, so every truncated factor carries
    at least Nt/2 powers of W. A sector or eigenstate is dropped only when
    (lambda/lambda_0)^(Nt/2) < tol.
    """
    n, Nt = model.n, model.Nt
    half = Nt // 2
    cut = math.log(tol) / half
    # 1) sector screening: an upper bound of the leading eigenvalue decides
    #    whether the exact leading eigenvalue has to be computed at all
    lead = {}
    t0 = time.time()
    bounds = {s: model.log_lambda_bound(*s) for s in model.sectors()}
    ll0 = -np.inf
    for s in sorted(model.sectors(), key=lambda s: -bounds[s]):
        if bounds[s] < ll0 + cut:
            continue            # ll0 only grows, so this sector can never contribute
        lead[s] = _lowlying_sector(model, *s, nev=1)["loglam"][0]
        ll0 = max(ll0, lead[s])
    keep = [s for s in lead if lead[s] >= ll0 + cut]
    if verbose:
        print(f"# lowlying: {len(keep)} of {len(bounds)} sectors contribute (tol={tol:g}, "
              f"{len(lead)} diagonalized); leading log lambda = {ll0:.10f} ({time.time()-t0:.1f}s)", flush=True)

    # 2) enough eigenpairs per sector so that the next one is below tol
    sec = {}
    for s in keep:
        nev = nev0
        while True:
            d = _lowlying_sector(model, *s, nev=nev)
            if d["full"] or d["loglam"][-1] < ll0 + cut:
                break
            nev *= 2
        m = d["loglam"] >= ll0 + cut
        sec[s] = dict(loglam=d["loglam"][m], Q=d["Q"][:, m], Dh=d["Dh"])
        if verbose:
            print(f"#   sector {s}: dim {d['Q'].shape[0]}, {m.sum()} eigenstates kept, "
                  f"log lambda_max - ll0 = {d['loglam'][0]-ll0:.4f}", flush=True)

    Z = sum(np.exp(Nt * (d["loglam"] - ll0)).sum() for d in sec.values())

    def X_apply(species, x, s, sp, v):     # v: (..., dim sp) -> (..., dim s)
        (a, b), (ap, bp) = s, sp
        if species == "particle":
            C = model.C[(x, ap)]
            return np.einsum("ij,...jb->...ib", C, v)          # (C x 1) v
        C = model.C[(x, b)]
        return v @ C                                           # (1 x C^T) v

    def Y_apply(species, y, s, sp, v):     # v: (..., dim s) -> (..., dim sp)
        (a, b), (ap, bp) = s, sp
        if species == "particle":
            C = model.C[(y, ap)]
            return np.einsum("ji,...jb->...ib", C, v)          # (C^T x 1) v
        C = model.C[(y, b)]
        return v @ C.T                                         # (1 x C) v

    out = {}
    for species in ("particle", "hole"):
        G = np.zeros((Nt, n, n))
        for s in model.sectors():
            sp = _partner(species, s)
            if not (0 <= sp[0] <= n and 0 <= sp[1] <= n):
                continue
            if s not in sec and sp not in sec:
                continue
            shp_s = (model.space.dim(s[0]), model.space.dim(s[1]))
            shp_p = (model.space.dim(sp[0]), model.space.dim(sp[1]))
            Dv_s = np.exp(-model.delta * model.HV(*s))
            Dv_p = np.exp(-model.delta * model.HV(*sp))

            # t <= Nt/2: expand in sector s;  Tr_s[W_s^(Nt-t) X W_sp^t Y]
            if s in sec:
                d = sec[s]
                for ks in _chunks(len(d["loglam"]), chunk):
                    Q = d["Q"][:, ks].T.reshape(-1, *shp_s)               # (k, da, db)
                    left = d["Dh"] * Q                                    # D^{1/2} q_k
                    right = Q / d["Dh"]                                   # D^{-1/2} q_k
                    # R[k, y] = Y_y right_k in sector sp ; Lx[k, x] = X_x^T left_k = Y_x left_k
                    R = np.stack([Y_apply(species, y, s, sp, right) for y in range(n)], axis=1)
                    Lx = np.stack([Y_apply(species, x, s, sp, left) for x in range(n)], axis=1)
                    for t in range(0, half + 1):
                        w = np.exp((Nt - t) * (d["loglam"][ks] - ll0))   # (k,)
                        G[t] += np.einsum("k,kxab,kyab->xy", w, Lx, R)
                        if t < half:
                            R = model.apply_W(*sp, R, Dv_p) * math.exp(-ll0)
            # t > Nt/2: expand in sector sp;  Tr_sp[W_sp^t Y W_s^(Nt-t) X]
            if sp in sec:
                d = sec[sp]
                for ks in _chunks(len(d["loglam"]), chunk):
                    Q = d["Q"][:, ks].T.reshape(-1, *shp_p)
                    left = d["Dh"] * Q
                    right = Q / d["Dh"]
                    # R[k, x] = X_x right_k in sector s ; Ly[k, y] = Y_y^T left_k = X_y left_k
                    R = np.stack([X_apply(species, x, s, sp, right) for x in range(n)], axis=1)
                    Ly = np.stack([X_apply(species, y, s, sp, left) for y in range(n)], axis=1)
                    for m_ in range(1, Nt - half):     # R holds W_s^m_ X right ;  t = Nt - m_ > Nt/2
                        R = model.apply_W(*s, R, Dv_s) * math.exp(-ll0)
                        t = Nt - m_
                        w = np.exp(t * (d["loglam"][ks] - ll0))
                        G[t] += np.einsum("k,kyab,kxab->xy", w, Ly, R)
        out[species] = G / Z
    return out


def _chunks(n, size):
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


# =============================================================================
# Fixed-field check of the operator mapping against an explicit fermion matrix
# =============================================================================


def fermion_matrix(K, phi, delta, species):
    """NSL's HubbardExp: M[t,t] = 1, M[t,t-1] = -B_t e^{s delta K} diag(e^{i s phi_{t-1}}), B_0 = -1."""
    Nt, n = phi.shape
    sgn = 1 if species == "particle" else -1
    eK = sla.expm(sgn * delta * K)
    M = np.eye(Nt * n, dtype=complex)
    for t in range(Nt):
        tm = (t - 1) % Nt
        B = -1.0 if t == 0 else 1.0
        M[t*n:(t+1)*n, tm*n:(tm+1)*n] -= B * eK @ np.diag(np.exp(1j * sgn * phi[tm]))
    return M


def self_test(n=3, Nt=4, seed=0):
    rng = np.random.default_rng(seed)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, (i+1) % n] = K[(i+1) % n, i] = 1.0
    K[0, 2] = K[2, 0] = 0.7       # non-bipartite on purpose: no hidden symmetry helps
    beta = 1.3
    delta = beta / Nt
    phi = rng.normal(size=(Nt, n))

    # full 2n-mode Fock space, modes: (x,up) = x, (x,dn) = n + x
    m = 2 * n
    dim = 2**m
    def c(j):
        C = np.zeros((dim, dim))
        for s in range(dim):
            if (s >> j) & 1:
                C[s ^ (1 << j), s] = -1.0 if bin(s & ((1 << j) - 1)).count("1") % 2 else 1.0
        return C
    cs = [c(j) for j in range(m)]
    num = [C.T @ C for C in cs]
    Khat = sum(K[x, y] * (cs[x].T @ cs[y] + cs[n+x].T @ cs[n+y]) for x in range(n) for y in range(n))
    q = [num[x] + num[n+x] - np.eye(dim) for x in range(n)]
    eK = sla.expm(delta * Khat)
    T = [eK @ sla.expm(1j * sum(phi[s, x] * q[x] for x in range(n))) for s in range(Nt)]

    def prod(lo, hi):   # T_{hi-1} ... T_lo
        P = np.eye(dim, dtype=complex)
        for s in range(lo, hi):
            P = T[s] @ P
        return P
    Z = np.trace(prod(0, Nt))
    Mp, Mh = fermion_matrix(K, phi, delta, "particle"), fermion_matrix(K, phi, delta, "hole")
    errs = [abs(Z - np.linalg.det(Mp) * np.linalg.det(Mh)) / abs(Z)]
    iMp, iMh = np.linalg.inv(Mp), np.linalg.inv(Mh)
    for t in range(Nt):
        for x in range(n):
            for y in range(n):
                gp = np.trace(prod(t, Nt) @ cs[x] @ prod(0, t) @ cs[y].T) / Z
                gh = np.trace(prod(t, Nt) @ cs[n+x].T @ prod(0, t) @ cs[n+y]) / Z
                errs += [abs(gp - iMp[t*n+x, y]), abs(gh - iMh[t*n+x, y])]
    err_map = max(errs)

    # dense vs lowlying backend on a small PPP model (ring of 4)
    K4 = np.zeros((4, 4))
    for i in range(4):
        K4[i, (i+1) % 4] = K4[(i+1) % 4, i] = 1.0
    pos = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    r = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    U = 8 / 2.4
    xx = U * 2.4 * 2.0 * 1.4 * r / E2_EV_ANGSTROM
    V = U / np.sqrt(1 + xx**2)
    model = PPP(K4, V, beta=6.0, Nt=24)
    gd = propagators_dense(model)
    gl = propagators_lowlying(model, tol=1e-14, nev0=4, verbose=False)
    err_backend = max(np.abs(gd[k] - gl[k]).max() for k in gd)

    # Trotter: lattice -> continuum as Nt grows
    gc = propagators_dense(PPP(K4, V, beta=2.0, Nt=8), continuum=True)
    d1 = propagators_dense(PPP(K4, V, beta=2.0, Nt=8))
    d2 = propagators_dense(PPP(K4, V, beta=2.0, Nt=16))
    gc2 = propagators_dense(PPP(K4, V, beta=2.0, Nt=16), continuum=True)
    e1 = np.abs(d1["particle"] - gc["particle"]).max()
    e2 = np.abs(d2["particle"] - gc2["particle"]).max()

    print(f"fixed-field mapping to NSL's fermion matrix: max error {err_map:.2e}")
    print(f"dense vs lowlying backend:                   max error {err_backend:.2e}")
    print(f"Trotter error Nt=8 -> 16: {e1:.3e} -> {e2:.3e} (ratio {e1/e2:.2f}, expect ~4 for O(delta^2))")
    ok = err_map < 1e-10 and err_backend < 1e-9
    print("SELF TEST", "PASSED" if ok else "FAILED")
    return ok


# =============================================================================
# CLI
# =============================================================================


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("yml", nargs="?", help="NSL yml file (system + PPP blocks)")
    p.add_argument("-o", "--output", help="output h5 file")
    p.add_argument("--backend", choices=["dense", "lowlying"], default=None,
                   help="default: dense for <= 8 sites, lowlying otherwise")
    p.add_argument("--continuum", action="store_true", help="also compute exp(-tau H) propagators (dense only)")
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not args.yml or not args.output:
        p.error("yml and --output are required")

    import yaml
    import h5py
    with open(args.yml) as f:
        yml = yaml.safe_load(f)
    K = hopping_matrix(yml["system"])
    V = interaction_matrix(yml)
    beta, Nt = float(yml["system"]["beta"]), int(yml["system"]["Nt"])
    if float(yml["system"].get("mu", 0) or 0) != 0:
        raise SystemExit("ppp_ed.py: only mu = 0 is implemented")
    model = PPP(K, V, beta, Nt)
    backend = args.backend or ("dense" if model.n <= 8 else "lowlying")
    print(f"# {yml['system']['name']}: n={model.n}, beta={beta}, Nt={Nt}, backend={backend}", flush=True)

    t0 = time.time()
    G = propagators_dense(model) if backend == "dense" else propagators_lowlying(model, tol=args.tol)
    print(f"# lattice propagators done in {time.time()-t0:.1f}s", flush=True)
    with h5py.File(args.output, "w") as f:
        f["K"], f["V"] = K, V
        f.attrs.update(beta=beta, Nt=Nt, backend=backend, tol=args.tol, name=str(yml["system"]["name"]))
        for sp in ("particle", "hole"):
            f[f"lattice/{sp}"] = G[sp]
        if args.continuum:
            if backend != "dense":
                raise SystemExit("--continuum needs the dense backend")
            Gc = propagators_dense(model, continuum=True)
            for sp in ("particle", "hole"):
                f[f"continuum/{sp}"] = Gc[sp]
    print(f"# wrote {args.output}")


if __name__ == "__main__":
    main()
