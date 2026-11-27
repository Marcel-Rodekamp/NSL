#!/usr/bin/env python3
"""Stage-1 reduction of a measured PPP (or Hubbard) ensemble: one chain file in, one small file out.

    python ppp_reduce.py chain.yml [-o reduced.h5] [--h5 chain.h5] [--orbitals N] [--bin B] [--single]
                         [--probe-slices 4,8,16,32] [--detsign-every K] [--max-configs N] [--overwrite]

Input: the yml file of the run and its h5 file (fileIO: h5file, relative to the working directory,
as for the NSL programs), with position-space propagators measured by Executables/Hubbard/propagator
(measurements: basis: charge, type: spatial). Only the particle species is read: for a fixed
auxiliary field the hole propagators are redundant (see ppp_wick.py); they are compared with the
particle ones on the first configurations as a check of the stored data.

Everything that needs products of propagators of the same configuration at the same time is
computed here (equal-time spin and charge correlations); what stays linear is stored per
configuration (--bin 1, the default) or averaged over bins of B consecutive configurations (--bin B; the
analysis can then only use bins of multiples of B).

Bridge link (chains generated with `HMC: bridge:`, notes deck 7): the configurations are distributed with
w(phi) Lambda(phi) instead of the target w(phi). Every per-configuration quantity O is multiplied by the weight
1/Lambda before it is binned, and the bins of the weight are stored in `weights`: the estimate of <O> is
sum_b O_b / sum_b weights_b (ppp_analysis does this). Lambda = 1/(2 Nx) sum_{i, x = +-1}
|1 + (e^{i theta x} - 1) n_i(b)|^2 with n_i(b) = 1 - G_ii(b, b) at the bridge slice b; for the multi-slice bridge
(`slice: all`) the same mean also runs over all time slices b. theta and b come from
<name>/Meta/bridge of the ensemble file (written by autotune_RadialMCMC_PPP) or `HMC: bridge` of the yml.
Without the bridge every weight is 1 and the stored bins are plain averages, as before. The Monte Carlo history (config/...) and the probes stay
per configuration, for autocorrelation times. Two-body (exciton, pp, hh) correlators need products of
propagators of different times and come with stage 2 (ppp_reduce2.py).

Output (plain h5py; c = configuration index, b = bin index (= c for --bin 1), t = Nt time separations,
sizes in the attributes):
  attrs                          name, Nt, beta, delta, Nx, mu, sources, n_configs, bin, n_bins, input_h5,
                                 input_yml_text, bridge (0/1), bridge_theta, bridge_slice (int, -1 without the
                                 bridge link, or the string "all" for the multi-slice bridge), ...
                                 (n_configs = n_bins * bin: a remainder is not reduced)
  weights                        (b,) bin averages of the weight 1/Lambda (all 1 without the bridge link); the
                                 binned datasets below (onebody, equaltime) are bin averages of weight * O
  config/bridge_lambda           (c,) Lambda of every configuration (bridge link only)
  lattice/{K, V, positions, sublattice, V_eigenvalues, V_eigenvectors}
  orbitals/U                     (Nx, No) hopping orbitals as columns: symmetry adapted (site permutations that
                                 leave K and V invariant), sorted by eps; on a bipartite lattice the orbital
                                 with -eps is (-1)^x times the one with +eps (sublattice sign)
  orbitals/{eps, class, partner} eps = -(lambda + mu); class index into orbitals/class_names;
                                 partner = index of the orbital with -eps (-1: none)
  config/{id, S, acceptanceProbability, markovTime}    as stored by the HMC
  config/{S_gauge, field_modes}  1/2 sum_t phi_t (delta V)^-1 phi_t and a_k = sum_t phi_t.v_k / sqrt(Nt delta lambda_k)
                                 (k = 0: largest eigenvalue of V), as in ppp_autocorr.py
  config/detsign                 sign of det M_p e^{-i sum phi/2} (real at mu = 0 on a bipartite lattice; +1 in
                                 the region of phi = 0); 0 where not computed. config/detsign_residual = |Im|.
                                 Computed from the stabilized transfer-matrix product (logdet_particle; agrees
                                 with the dense determinant of ppp_ed.fermion_matrix to 1e-14, ~15 ms per config)
  config/S_fermion_phi           -log|det M_p|^2 = -2 Re log det M_p at the stored field phi, from the same product
                                 (mu = 0, bipartite; NaN where not computed). Without the bridge link it equals
                                 S - S_gauge (checks/attrs S_fermion_phi_max_dev). With it, S of the HMC holds the
                                 fermion action at the shifted field phi + c_k, so S - S_gauge is a quantity of the
                                 sampled ensemble, and S_gauge + S_fermion_phi is the action of the target ensemble
                                 (to be reweighted with 1/Lambda like every other observable). Bridge-link runs
                                 compute it, and the sign, on every configuration, whatever --detsign-every.
  onebody/<class>/forward       (b, t, n, n) sum over sources s of sign * U^T G(s+t, s) U / #sources,
                                 restricted to the orbitals of the class (the ensemble average is block diagonal)
  onebody/<class>/backward       (b, t, n, n) the same for G(s, s+t); sign = -1 where s+t wraps around Nt
  onebody/<class>/orbitals       indices of the class orbitals in orbitals/U
  onebody/<class>/addition_probe (c, p, n) per configuration, not weighted: real part of the diagonal addition correlator
                                 A_aa(t) = [F_aa(t) - B_aa(Nt - t)] / 2 (t > 0; F_aa(0) at t = 0) at the time
                                 slices onebody/attrs/probe_slices (for tau_int of the correlators)
  equaltime/{G, ss, qq}          (b, Nx, Nx) site basis, averaged over the Nt time slices: G(t,t) = <c c^+>,
                                 <s_x s_y> and <q_x q_y> (s = n_up - n_down, q = n_up + n_down - 1)
  noninteracting/...             the same quantities for phi = 0 (if measured), without the config axis (double)
  checks/attrs                   hole_relation_max_dev, ...

The file is written as <out>.tmp and renamed when complete; an existing output is skipped unless
--overwrite is given.
"""

import argparse
import datetime
import os
import sys
import time
from collections import deque

import numpy as np
import scipy.linalg as sla
import h5py
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppp_ed  # noqa: E402
import ppp_wick as wick  # noqa: E402

VERSION = 4          # 2: --bin, addition_probe; 3: bridge-link weights; 4: config/S_fermion_phi


# ----------------------------------------------------------------------------------------------
# lattice, symmetry and orbitals
# ----------------------------------------------------------------------------------------------

def interaction(yml, n):
    if "PPP" in yml:
        return ppp_ed.interaction_matrix(yml)
    return float(yml["system"]["U"]) * np.eye(n)          # Hubbard programs: on-site U only


def sublattice(K):
    """+-1 per site from a two-colouring of the hopping graph, or None if it is not bipartite"""
    n = len(K)
    col = np.zeros(n, dtype=int)
    for s in range(n):
        if col[s]:
            continue
        col[s] = 1
        queue = deque([s])
        while queue:
            x = queue.popleft()
            for y in np.nonzero(K[x])[0]:
                if y == x:
                    continue
                if col[y] == 0:
                    col[y] = -col[x]
                    queue.append(y)
                elif col[y] == col[x]:
                    return None
    return col.astype(float)


def site_permutation(pos, f, tol=1e-6):
    q = f(pos)
    perm = np.empty(len(pos), dtype=int)
    for i in range(len(pos)):
        d = np.linalg.norm(pos - q[i], axis=1)
        j = int(np.argmin(d))
        if d[j] > tol:
            return None
        perm[i] = j
    if len(set(perm)) != len(pos):
        return None
    P = np.zeros((len(pos), len(pos)))
    P[perm, np.arange(len(pos))] = 1.0
    return P


def symmetry_operations(pos, K, V):
    """Commuting reflections (site permutations) that leave K and V invariant.

    Mirrors through the centroid along the coordinate axes; if none is found, along the
    principal axes of the positions. Returns a list of (name, P)."""
    pos = np.asarray(pos, float)
    pos = pos - pos.mean(axis=0)
    frames = [np.eye(pos.shape[1])]
    if pos.shape[1] >= 2:
        w, R = np.linalg.eigh(np.cov(pos.T))
        frames.append(R[:, ::-1])
    for R in frames:
        p = pos @ R
        ops = []
        for ax in range(min(2, p.shape[1])):
            s = np.ones(p.shape[1])
            s[ax] = -1
            P = site_permutation(p, lambda x, s=s: x * s)
            if P is not None and np.allclose(P @ K @ P.T, K) and np.allclose(P @ V @ P.T, V):
                ops.append((f"mirror {'xyz'[ax]}", P))
        if ops:
            return ops
    return []


def orbital_basis(K, V, pos, mu=0.0, tol=1e-8):
    """Symmetry-adapted hopping orbitals, sorted by eps = -(lambda + mu); see the module doc."""
    n = len(K)
    lam, W = np.linalg.eigh(K)
    ops = symmetry_operations(pos, K, V)
    chars = [np.array(c) for c in np.ndindex(*([2] * len(ops)))]           # 0 -> +1, 1 -> -1
    chars = [1 - 2 * c for c in chars]
    projs = []
    for c in chars:
        Pc = np.eye(n)
        for (name, P), s in zip(ops, c):
            Pc = Pc @ (np.eye(n) + s * P) / 2
        projs.append(Pc)
    names = ["".join("+" if s > 0 else "-" for s in c) for c in chars] if ops else ["all"]
    if not ops:
        projs = [np.eye(n)]
    r2 = np.sum((np.asarray(pos, float) - np.mean(pos, axis=0)) ** 2, axis=1)

    def fix_sign(u):
        k = int(np.argmax(np.abs(u) > np.abs(u).max() - 1e-9))
        return u if u[k] > 0 else -u

    def adapt(Wc):
        """symmetry-adapted, deterministic orthonormal basis of the space spanned by Wc"""
        vecs, cls = [], []
        for ci, Pc in enumerate(projs):
            M = Pc @ Wc
            u, s, _ = np.linalg.svd(M, full_matrices=False)
            B = u[:, s > 1e-8]
            if B.shape[1] == 0:
                continue
            if B.shape[1] > 1:                                  # fix the basis inside the subspace
                _, ev = np.linalg.eigh(B.T @ np.diag(r2) @ B)
                B = B @ ev
            for k in range(B.shape[1]):
                vecs.append(fix_sign(B[:, k]))
                cls.append(ci)
        assert len(vecs) == Wc.shape[1], "symmetry adaptation lost or gained vectors"
        return vecs, cls

    clusters, start = [], 0
    for k in range(1, n + 1):
        if k == n or abs(lam[k] - lam[start]) > 1e-8 * max(1.0, abs(lam[start])):
            clusters.append(list(range(start, k)))
            start = k
    U, cls, eps = [], [], []
    for cl in clusters:
        vecs, cc = adapt(W[:, cl])
        U += vecs
        cls += cc
        eps += [-(lam[cl[0]] + mu)] * len(vecs)
    U, cls, eps = np.array(U).T, np.array(cls), np.array(eps)

    partner = -np.ones(n, dtype=int)
    sub = sublattice(K)
    if sub is not None and abs(mu) < tol:
        # valence orbitals := sublattice partners of the conduction orbitals (u_{-p} = (-1)^x u_p)
        cond = np.nonzero(eps > tol)[0]
        val = np.nonzero(eps < -tol)[0]
        if len(cond) == len(val):
            # orbitals with eps >= 0 stay; the valence ones are replaced by the partners of the conduction ones
            keep_idx = [k for k in range(n) if eps[k] >= -tol]
            cols = [U[:, k] for k in keep_idx] + [sub * U[:, i] for i in cond]
            new_eps = [eps[k] for k in keep_idx] + [-eps[i] for i in cond]
            new_cls = [cls[k] for k in keep_idx] + [int(np.argmax([np.linalg.norm(Pc @ (sub * U[:, i])) for Pc in projs]))
                                                     for i in cond]
            pair_of = {len(keep_idx) + j: keep_idx.index(i) for j, i in enumerate(cond)}   # new valence -> its conduction
            order = np.argsort(np.array(new_eps), kind="stable")
            inv = np.empty(n, dtype=int)
            inv[order] = np.arange(n)
            U, eps, cls = np.array(cols).T[:, order], np.array(new_eps)[order], np.array(new_cls)[order]
            for v, cnd in pair_of.items():
                partner[inv[v]], partner[inv[cnd]] = inv[cnd], inv[v]
        else:
            print("# warning: unequal numbers of positive and negative eps; no partner orbitals")
    order = np.argsort(eps, kind="stable")
    if not np.all(order == np.arange(n)):
        inv = np.empty(n, dtype=int)
        inv[order] = np.arange(n)
        U, cls, eps = U[:, order], cls[order], eps[order]
        partner = np.where(partner[order] >= 0, inv[partner[order]], -1)

    # checks
    assert np.allclose(U.T @ U, np.eye(n), atol=1e-10), "orbitals not orthonormal"
    assert np.allclose(U.T @ K @ U, np.diag(-(eps + mu)), atol=1e-9), "orbitals do not diagonalize K"
    for k in range(n):
        assert np.linalg.norm(projs[cls[k]] @ U[:, k]) > 1 - 1e-8, "orbital not in a single class"
        if partner[k] >= 0:
            assert np.allclose(U[:, partner[k]], sub * U[:, k]), "partner is not the sublattice image"
    if sub is not None and np.any(np.abs(eps) < tol):
        print("# warning: zero modes; they have no partner orbitals")
    return dict(U=U, eps=eps, cls=cls, class_names=names, partner=partner, sublattice=sub,
                operations=[name for name, _ in ops])


# ----------------------------------------------------------------------------------------------
# fermion determinant
# ----------------------------------------------------------------------------------------------

def logdet_particle(K, phi, delta, eK=None):
    """complex log det M_p of NSL's HubbardExp particle matrix (= ppp_ed.fermion_matrix), from the
    transfer matrices: det M_p = det(1 + b_0 b_{Nt-1} ... b_1), b_t = e^{delta K} diag(e^{i phi_{t-1}}),
    with the product kept as Q D T (pivoted QR at every step; D splits into large and small parts)."""
    Nt, n = phi.shape
    if eK is None:
        eK = sla.expm(delta * K)
    Q, D, T = np.eye(n, dtype=complex), np.ones(n), np.eye(n, dtype=complex)
    for t in list(range(1, Nt)) + [0]:
        b = eK * np.exp(1j * phi[(t - 1) % Nt])[None, :]
        Qn, R, piv = sla.qr((b @ Q) * D[None, :], pivoting=True)
        d = np.abs(np.diag(R))
        # (b Q D)[:, piv] = Qn R, so b Q D T = Qn diag(d) (R/d) T[piv]
        T, Q, D = (R / d[:, None]) @ T[piv], Qn, d
    Db, Ds = np.maximum(D, 1.0), np.minimum(D, 1.0)
    X = (Q.conj().T @ np.linalg.inv(T)) / Db[:, None] + np.diag(Ds)
    out = np.sum(np.log(Db))
    with np.errstate(divide="ignore", invalid="ignore"):   # spurious LAPACK warnings (Accelerate)
        for M in (Q, T, X):
            s, l = np.linalg.slogdet(M)
            out = out + l + np.log(s)
    return out


# ----------------------------------------------------------------------------------------------
# reading NSL data
# ----------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------
# bridge link (notes deck 7; src/NSL/MarkovChain/bridgeLink.tpp)
# ----------------------------------------------------------------------------------------------

ALL_SLICES = "all"      # the bridge slice of the multi-slice bridge (all time slices)


def bridge_slice(v):
    """a bridge slice from the yml, an HDF5 attribute or dataset: an int, or ALL_SLICES (the string all)"""
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v[()]
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str) and v.strip() == ALL_SLICES:
        return ALL_SLICES
    return int(v)


def bridge_settings(yml, meta=None, warn=True):
    """(theta, slice) of a chain generated with the bridge link, or None; slice is an int or ALL_SLICES.

    Sources: `meta`, the group <name>/Meta/bridge of the ensemble file written by autotune_RadialMCMC_PPP (None if
    absent), and `HMC: bridge:` of the yml (defaults theta = pi/2, slice 0, as in the driver). The record in the
    ensemble file wins; a yml that disagrees with it is an error, because it is not the yml of this chain.
    warn=False: no warning for a bridge in the yml without a record (an empty file written before the chain)."""
    hmc = yml.get("HMC") or {}
    from_yml = None
    if "bridge" in hmc:
        b = hmc["bridge"] or {}
        from_yml = (float(b.get("theta", np.pi / 2)), bridge_slice(b.get("slice", 0)))
    from_h5 = None if meta is None else (float(np.real(meta["theta"][()])), bridge_slice(meta["slice"][()]))
    if from_h5 is not None:
        if from_yml is None or abs(from_yml[0] - from_h5[0]) > 1e-12 or from_yml[1] != from_h5[1]:
            raise SystemExit(f"the ensemble was generated with the bridge link theta = {from_h5[0]}, slice "
                             f"{from_h5[1]} (Meta/bridge), but the yml says {from_yml or 'no bridge'}")
        return from_h5
    if from_yml is not None and warn:
        print(f"# warning: `HMC: bridge` in the yml but no Meta/bridge in the ensemble file (generated before it "
              f"was recorded?); using theta = {from_yml[0]}, slice {from_yml[1]} from the yml")
    return from_yml


def bridge_block(Gtt, slc):
    """the equal-time propagators that enter Lambda: G(b, b) (Nx, Nx) at the bridge slice, or all of Gtt (Nt, Nx, Nx)
    for the multi-slice bridge"""
    return Gtt if slc == ALL_SLICES else Gtt[slc]


def bridge_lambda(Gbb, theta):
    """Lambda = 1/(2 Nx) sum_{i, x = +-1} |1 + (e^{i theta x} - 1) n_i|^2 of one configuration, n_i = 1 - G_ii(b, b)
    the generalized density at the bridge slice (G = M_p^-1, particles), and max_i |Re n_i - 1/2| / max(1, |n_i|),
    which vanishes up to rounding at mu = 0 on a bipartite lattice (a check of the conventions; relative, because
    |n| grows like 1/det near a zero of the determinant). Gbb (Nt, Nx, Nx), all equal-time propagators: the
    multi-slice bridge, the mean also over the time slices (bridge_block)."""
    n = 1.0 - np.diagonal(Gbb, axis1=-2, axis2=-1)
    lam = 0.5 * np.mean(np.abs(1 + (np.exp(1j * theta) - 1) * n) ** 2 + np.abs(1 + (np.exp(-1j * theta) - 1) * n) ** 2)
    return float(lam), float(np.max(np.abs(n.real - 0.5) / np.maximum(1.0, np.abs(n))))


# ----------------------------------------------------------------------------------------------
# what does not depend on the configurations (shared by main and write_skeleton)
# ----------------------------------------------------------------------------------------------

def static_setup(yml, n_orbitals=0):
    """Lattice, interaction, orbital basis and orbital classes of a run: everything of the reduction that does not
    depend on the configurations. n_orbitals > 0 keeps only that many orbitals closest to the Fermi level."""
    sysyml = yml["system"]
    Nt, beta = int(sysyml["Nt"]), float(sysyml["beta"])
    Nx, mu = int(sysyml["nions"]), float(sysyml.get("mu", 0.0))
    offset = float(sysyml.get("offset", 0.0))
    K = ppp_ed.hopping_matrix(sysyml)
    V = interaction(yml, Nx)
    pos = np.array(sysyml["positions"], float)
    orb = orbital_basis(K, V, pos, mu)
    keep = np.arange(Nx)
    if 0 < n_orbitals < Nx:
        keep = np.sort(np.argsort(np.abs(orb["eps"]), kind="stable")[:n_orbitals])
    classes = {}
    for k in keep:
        classes.setdefault(orb["class_names"][orb["cls"][k]], []).append(int(k))
    lamV, QV = np.linalg.eigh(V)
    order = np.argsort(lamV)[::-1]
    do_sign = abs(mu) < 1e-12 and abs(offset) < 1e-12 and orb["sublattice"] is not None
    return dict(name=sysyml["name"], Nt=Nt, beta=beta, delta=beta / Nt, Nx=Nx, mu=mu, offset=offset, K=K, V=V,
                pos=pos, orb=orb, keep=keep, classes=classes, lamV=lamV[order], QV=QV[:, order], do_sign=do_sign)


def write_static(fo, st, **attrs):
    """The attributes (those of static_setup and `attrs`) and the groups lattice/ and orbitals/ of a stage-1 file"""
    orb = st["orb"]
    fo.attrs.update(dict(name=st["name"], Nt=st["Nt"], beta=st["beta"], delta=st["delta"], Nx=st["Nx"], mu=st["mu"],
                         offset=st["offset"], **attrs))
    fo["lattice/K"], fo["lattice/V"], fo["lattice/positions"] = st["K"], st["V"], st["pos"]
    fo["lattice/V_eigenvalues"], fo["lattice/V_eigenvectors"] = st["lamV"], st["QV"]
    if orb["sublattice"] is not None:
        fo["lattice/sublattice"] = orb["sublattice"]
    fo["orbitals/U"], fo["orbitals/eps"], fo["orbitals/class"] = orb["U"], orb["eps"], orb["cls"]
    fo["orbitals/partner"] = orb["partner"]
    fo["orbitals/class_names"] = np.array(orb["class_names"], dtype="S")
    fo["orbitals/symmetry_operations"] = np.array(orb["operations"] or ["none"], dtype="S")
    fo["orbitals/kept"] = st["keep"]


def source_times(Nt, n):
    """The source times of Executables/Hubbard/propagator for `Number Time Sources: n`: 0, step, 2 step, ... < Nt with
    step = ceil(Nt / n) (FermionPropagator::sourceTimes)"""
    step = -(-Nt // max(1, n))
    return list(range(0, Nt, step))


def write_skeleton(yml_text, yml, out, B=1, single=False, probe_slices="4,8,16,32", n_orbitals=0, h5in=None,
                   yml_path=None):
    """An empty stage-1 file for the measurement on the fly (src/NSL/Measurements/Hubbard/reducedObservables.tpp).

    The same attributes, lattice/ and orbitals/ as main() writes (n_configs = n_bins = 0), and every per-configuration
    and per-bin dataset with length 0, extendible along its first axis. The C++ appends complete bins of B
    configurations, then updates n_configs, n_bins and checks/attrs; it also writes noninteracting/ (the propagators
    at phi = 0). The sources are those of `measurements: Number Time Sources` (default 1, as for the propagator
    program), the bridge link that of `HMC: bridge` (the C++ checks it against Meta/bridge of the chain)."""
    st = static_setup(yml, n_orbitals)
    Nt, Nx, classes = st["Nt"], st["Nx"], st["classes"]
    meas = yml.get("measurements") or {}
    sources = source_times(Nt, int(meas.get("Number Time Sources", 1)))
    bridge = bridge_settings(yml, None, warn=False)
    if bridge is not None and not st["do_sign"]:
        raise SystemExit("the bridge link needs mu = 0, offset = 0 and a bipartite lattice")
    probe = np.array(sorted({int(x) for x in probe_slices.split(",") if x.strip() and 0 <= int(x) < Nt}), int)
    ctype = np.complex64 if single else np.complex128
    rtype = np.float32 if single else np.float64
    h5in = h5in or yml["fileIO"]["h5file"]
    tmp = out + ".tmp"
    with h5py.File(tmp, "w") as fo:
        write_static(fo, st, sources=np.array(sources), n_configs=0, input_h5=os.path.abspath(h5in),
                     input_yml=os.path.abspath(yml_path) if yml_path else "", input_yml_text=yml_text,
                     version=VERSION, on_the_fly=1,
                     created=datetime.datetime.now().isoformat(timespec="seconds"),
                     precision="single" if single else "double", bin=B, n_bins=0,
                     bridge=int(bridge is not None), bridge_theta=bridge[0] if bridge else np.nan,
                     bridge_slice=bridge[1] if bridge else -1)

        def grow(path, shape, dtype, chunk=256):        # length 0, extendible along the first axis
            return fo.create_dataset(path, (0,) + tuple(shape), dtype, maxshape=(None,) + tuple(shape),
                                     chunks=(chunk,) + tuple(shape))
        grow("config/id", (), np.int64)
        grow("config/S", (), complex)
        for key in ("acceptanceProbability", "markovTime", "S_gauge", "detsign_residual"):
            grow(f"config/{key}", (), float)
        grow("config/field_modes", (Nx,), float)
        grow("config/detsign", (), np.int8)
        if st["do_sign"]:
            grow("config/S_fermion_phi", (), float)
        if bridge:
            grow("config/bridge_lambda", (), float)
        grow("weights", (), float)
        for c, idx in classes.items():
            m = len(idx)
            fo[f"onebody/{c}/orbitals"] = np.array(idx)
            grow(f"onebody/{c}/forward", (Nt, m, m), ctype, chunk=1)
            grow(f"onebody/{c}/backward", (Nt, m, m), ctype, chunk=1)
            grow(f"onebody/{c}/addition_probe", (len(probe), m), rtype)
        fo["onebody"].attrs["probe_slices"] = probe
        for key in ("G", "ss", "qq"):
            grow(f"equaltime/{key}", (Nx, Nx), ctype, chunk=1)
        chk = fo.create_group("checks")
        chk.attrs["hole_relation_max_dev"] = 0.0
        chk.attrs["hole_relation_configs"] = 0
        chk.attrs["detsign_max_residual"] = 0.0 if st["do_sign"] else np.nan
        chk.attrs["bridge_re_n_max_dev"] = 0.0 if bridge else np.nan
        chk.attrs["S_fermion_phi_max_dev"] = 0.0 if st["do_sign"] and not bridge else np.nan
    os.replace(tmp, out)
    print(f"wrote the empty stage-1 file {out}: {st['name']}, Nt = {Nt}, Nx = {Nx}, sources {sources}, bins of {B}, "
          f"orbital classes {', '.join(f'{c} ({len(v)})' for c, v in classes.items())}"
          + (f"; bridge link theta = {bridge[0]:.6g}, slice {bridge[1]}" if bridge else ""))
    return 0


def read_tensor(ds):
    return ds[()].reshape(tuple(ds.attrs["shape"]))


def sources_of(group):
    out = []
    for key in group.keys():
        if key.startswith("invM[t,") and key != "invM[t,t]":
            out.append(int(key[len("invM[t,"):-1]))
    return sorted(out)


def onebody(group, sources, Nt):
    """source-averaged forward G(s+t, s) and backward G(s, s+t), antiperiodic, and G(t, t)"""
    F = 0.0
    B = 0.0
    tt = np.arange(Nt)
    for s in sources:
        idx = (s + tt) % Nt
        sign = np.where(s + tt >= Nt, -1.0, 1.0)[:, None, None]
        F = F + sign * read_tensor(group[f"invM[t,{s}]"])[idx]
        B = B + sign * read_tensor(group[f"invM[{s},t]"])[idx]
    return F / len(sources), B / len(sources), read_tensor(group["invM[t,t]"])


def hole_relation_deviation(group_p, group_h, sources, Nt):
    """max relative deviation from G_h(s,t)^T = -G_p(t,s) (t > s) and G_h(t,t)^T = 1 - G_p(t,t)"""
    worst = 0.0
    for s in sources:
        col_p = read_tensor(group_p[f"invM[t,{s}]"])        # G_p(t, s)
        row_h = read_tensor(group_h[f"invM[{s},t]"])        # G_h(s, t)
        t = np.arange(s + 1, Nt)
        if len(t):
            a, b = np.swapaxes(row_h[t], 1, 2), -col_p[t]
            worst = max(worst, np.abs(a - b).max() / np.abs(b).max())
    Gp, Gh = read_tensor(group_p["invM[t,t]"]), read_tensor(group_h["invM[t,t]"])
    ref = np.eye(Gp.shape[-1]) - Gp
    worst = max(worst, np.abs(np.swapaxes(Gh, 1, 2) - ref).max() / np.abs(ref).max())
    return float(worst)


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("yml")
    p.add_argument("-o", "--output", help="default: reduced/<yml name>.h5")
    p.add_argument("--h5", help="ensemble file (default: fileIO: h5file of the yml)")
    p.add_argument("--orbitals", type=int, default=0,
                   help="keep only the N orbitals closest to the Fermi level (default: all)")
    p.add_argument("--bin", type=int, default=1,
                   help="average correlators and equal-time data over B consecutive configurations [1]")
    p.add_argument("--single", action="store_true", help="store correlators in single precision")
    p.add_argument("--probe-slices", default="4,8,16,32",
                   help="time slices of the per-configuration addition probes [4,8,16,32]")
    p.add_argument("--detsign-every", type=int, default=1, help="compute the determinant sign on every K-th config")
    p.add_argument("--max-configs", type=int, default=0, help="reduce only the first N configurations (tests)")
    p.add_argument("--check-hole", type=int, default=2, help="configurations used to check the hole relation")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skeleton", action="store_true",
                   help="write only the empty file for the measurement on the fly (write_skeleton); no ensemble needed")
    args = p.parse_args(argv)

    with open(args.yml) as fy:
        yml_text = fy.read()
    yml = yaml.safe_load(yml_text)
    h5in = args.h5 or yml["fileIO"]["h5file"]
    out = args.output or os.path.join("reduced", os.path.splitext(os.path.basename(args.yml))[0] + ".h5")
    if os.path.exists(out) and not args.overwrite:
        print(f"{out} exists; skipped (use --overwrite to redo)")
        return 0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if args.skeleton:
        return write_skeleton(yml_text, yml, out, max(1, args.bin), args.single, args.probe_slices, args.orbitals, h5in,
                              args.yml)

    st = static_setup(yml, args.orbitals)
    name, Nt, beta, delta, Nx, mu, offset = (st[k] for k in ("name", "Nt", "beta", "delta", "Nx", "mu", "offset"))
    K, V, pos, orb, keep, classes = (st[k] for k in ("K", "V", "pos", "orb", "keep", "classes"))
    lamV, QV, do_sign = st["lamV"], st["QV"], st["do_sign"]
    U = orb["U"]
    Ainv = np.linalg.inv(delta * V)
    ctype = np.complex64 if args.single else np.complex128
    eK = sla.expm(delta * K)

    t_start = time.time()
    with h5py.File(h5in, "r") as fin:
        chain = fin[f"{name}/markovChain"]
        ids = sorted(int(k) for k in chain.keys() if k.isdigit())
        measured = [i for i in ids if f"{i}/propagator/spatial/particle" in chain]
        if len(measured) < len(ids):
            print(f"# warning: propagators measured on {len(measured)} of {len(ids)} configurations; "
                  f"only those are reduced")
        ids = measured[: args.max_configs] if args.max_configs > 0 else measured
        if not ids:
            raise SystemExit(f"no configuration with propagator/spatial/particle in {h5in}")
        B = max(1, args.bin)
        nb = len(ids) // B
        if nb == 0:
            raise SystemExit(f"{len(ids)} configurations are fewer than one bin of {B}")
        if nb * B < len(ids):
            print(f"# {len(ids) - nb * B} configurations after the last complete bin of {B} are not reduced")
        ids = ids[: nb * B]
        sources = sources_of(chain[f"{ids[0]}/propagator/spatial/particle"])
        nc = len(ids)
        bridge = bridge_settings(yml, fin.get(f"{name}/Meta/bridge"))
        if bridge is not None and not do_sign:
            raise SystemExit("the bridge link needs mu = 0, offset = 0 and a bipartite lattice")
        probe = np.array(sorted({int(x) for x in args.probe_slices.split(",") if x.strip() and 0 <= int(x) < Nt}), int)
        print(f"{h5in}: {name}, Nt = {Nt}, Nx = {Nx}, {nc} configurations"
              f"{f' in {nb} bins of {B}' if B > 1 else ''}, sources {sources}, "
              f"orbital classes {', '.join(f'{c} ({len(v)})' for c, v in classes.items())}"
              f"{' (symmetry: ' + ', '.join(orb['operations']) + ')' if orb['operations'] else ''}"
              + (f"; bridge link theta = {bridge[0]:.6g}, slice {bridge[1]}: weights 1/Lambda" if bridge else ""))

        tmp = out + ".tmp"
        with h5py.File(tmp, "w") as fo:
            write_static(fo, st, sources=np.array(sources), n_configs=nc, input_h5=os.path.abspath(h5in),
                         input_yml=os.path.abspath(args.yml), input_yml_text=yml_text, version=VERSION,
                         created=datetime.datetime.now().isoformat(timespec="seconds"),
                         precision="single" if args.single else "double", bin=B, n_bins=nb,
                         bridge=int(bridge is not None), bridge_theta=bridge[0] if bridge else np.nan,
                         bridge_slice=bridge[1] if bridge else -1)

            cfg = fo.create_group("config")
            cfg["id"] = np.array(ids)
            dS = cfg.create_dataset("S", (nc,), complex)
            dP = cfg.create_dataset("acceptanceProbability", (nc,), float)
            dT = cfg.create_dataset("markovTime", (nc,), float)
            dSg = cfg.create_dataset("S_gauge", (nc,), float)
            dA = cfg.create_dataset("field_modes", (nc, Nx), float)
            dSign = cfg.create_dataset("detsign", (nc,), np.int8)
            dRes = cfg.create_dataset("detsign_residual", (nc,), float)
            dSF = cfg.create_dataset("S_fermion_phi", (nc,), float, fillvalue=np.nan) if do_sign else None
            sf_dev = 0.0                               # max |S - S_gauge - S_fermion_phi| (without the bridge link)
            dLam = cfg.create_dataset("bridge_lambda", (nc,), float) if bridge else None
            dW = fo.create_dataset("weights", (nb,), float)
            re_n_dev, wsum = 0.0, 0.0
            ob, prb = {}, {}
            rtype = np.float32 if args.single else np.float64
            for c, idx in classes.items():
                g = fo.create_group(f"onebody/{c}")
                g["orbitals"] = np.array(idx)
                m = len(idx)
                ob[c] = (g.create_dataset("forward", (nb, Nt, m, m), ctype, chunks=(1, Nt, m, m)),
                         g.create_dataset("backward", (nb, Nt, m, m), ctype, chunks=(1, Nt, m, m)))
                prb[c] = g.create_dataset("addition_probe", (nc, len(probe), m), rtype)
            fo["onebody"].attrs["probe_slices"] = probe
            eqG = fo.create_dataset("equaltime/G", (nb, Nx, Nx), ctype)
            eqS = fo.create_dataset("equaltime/ss", (nb, Nx, Nx), ctype)
            eqQ = fo.create_dataset("equaltime/qq", (nb, Nx, Nx), ctype)
            back = (Nt - probe) % Nt                   # B(Nt - t); unused at t = 0
            acc = None                                 # sums over the configurations of the current bin

            def project(M):                    # U^T M U for (Nt, Nx, Nx)
                return U.T @ M @ U

            # phi = 0 (NonInteracting), if measured
            ni = f"{name}/NonInteracting/propagator/spatial/particle"
            if ni in fin:
                gi = fin[ni]
                F, Bk, Gtt = onebody(gi, sources_of(gi), Nt)
                Fo, Bo = project(F), project(Bk)
                for c, idx in classes.items():
                    fo[f"noninteracting/onebody/{c}/forward"] = Fo[:, idx][:, :, idx]      # double: one entry
                    fo[f"noninteracting/onebody/{c}/backward"] = Bo[:, idx][:, :, idx]
                _, ss, qq = wick.equal_time(Gtt)
                fo["noninteracting/equaltime/G"] = Gtt.mean(axis=0)
                fo["noninteracting/equaltime/ss"] = ss.mean(axis=0)
                fo["noninteracting/equaltime/qq"] = qq.mean(axis=0)

            hole_dev = 0.0
            nsign = {1: 0, -1: 0}
            last = time.time()
            for n, i in enumerate(ids):
                g = chain[str(i)]
                phi = read_tensor(g["phi"])
                dS[n] = complex(np.asarray(g["actVal"][()]))
                dP[n] = float(np.real(np.asarray(g["acceptanceProbability"][()]))) if "acceptanceProbability" in g else np.nan
                dT[n] = float(np.real(np.asarray(g["markovTime"][()]))) if "markovTime" in g else np.nan
                pr = phi.real
                dSg[n] = 0.5 * np.einsum("tx,xy,ty->", pr, Ainv, pr)
                dA[n] = pr.sum(axis=0) @ QV / np.sqrt(Nt * delta * lamV)
                if do_sign and (n % args.detsign_every == 0 or bridge):
                    logdet = logdet_particle(K, phi, delta, eK)
                    r = np.exp(1j * logdet.imag - 0.5j * pr.sum())
                    if np.isfinite(logdet) and np.isfinite(r):
                        dSign[n] = 1 if r.real > 0 else -1
                        dRes[n] = abs(r.imag)
                        nsign[int(dSign[n])] += 1
                        dSF[n] = -2.0 * logdet.real
                        if not bridge:
                            sf_dev = max(sf_dev, abs(dS[n].real - dSg[n] - dSF[n]))
                    else:
                        dRes[n] = np.nan
                gp = g["propagator/spatial/particle"]
                F, Bk, Gtt = onebody(gp, sources, Nt)
                Fo, Bo = project(F), project(Bk)
                _, ss, qq = wick.equal_time(Gtt)
                w = 1.0
                if bridge:
                    lam, dev = bridge_lambda(bridge_block(Gtt, bridge[1]), bridge[0])
                    dLam[n], w, re_n_dev = lam, 1.0 / lam, max(re_n_dev, dev)
                wsum = w if n % B == 0 else wsum + w
                cur = {("eq", 0): w * Gtt.mean(axis=0), ("eq", 1): w * ss.mean(axis=0), ("eq", 2): w * qq.mean(axis=0)}
                for c, idx in classes.items():
                    Fc, Bc = Fo[:, idx][:, :, idx], Bo[:, idx][:, :, idx]
                    cur[(c, 0)], cur[(c, 1)] = w * Fc, w * Bc
                    dF = np.real(np.diagonal(Fc[probe], axis1=1, axis2=2))
                    dB = np.real(np.diagonal(Bc[back], axis1=1, axis2=2))
                    prb[c][n] = np.where((probe > 0)[:, None], 0.5 * (dF - dB), dF)
                acc = cur if n % B == 0 else {k: acc[k] + v for k, v in cur.items()}
                if n % B == B - 1:
                    b = n // B
                    for c in classes:
                        ob[c][0][b], ob[c][1][b] = acc[(c, 0)] / B, acc[(c, 1)] / B
                    eqG[b], eqS[b], eqQ[b] = (acc[("eq", k)] / B for k in range(3))
                    dW[b] = wsum / B
                if n < args.check_hole and "propagator/spatial/hole" in g:
                    hole_dev = max(hole_dev, hole_relation_deviation(gp, g["propagator/spatial/hole"], sources, Nt))
                if time.time() - last > 60 or n == nc - 1:
                    last = time.time()
                    print(f"  {n + 1}/{nc} configurations, {last - t_start:.0f} s", flush=True)
            chk = fo.create_group("checks")
            chk.attrs["hole_relation_max_dev"] = hole_dev
            chk.attrs["hole_relation_configs"] = min(args.check_hole, nc)
            chk.attrs["detsign_max_residual"] = float(np.max(dRes[()])) if do_sign else np.nan
            chk.attrs["bridge_re_n_max_dev"] = re_n_dev if bridge else np.nan
            chk.attrs["S_fermion_phi_max_dev"] = sf_dev if do_sign and not bridge else np.nan
    os.replace(tmp, out)
    size = os.path.getsize(out) / 2**20
    print(f"wrote {out} ({size:.0f} MB) in {time.time() - t_start:.0f} s")
    print(f"  hole relation (first {min(args.check_hole, nc)} configs): max deviation {hole_dev:.1e}")
    if do_sign:
        which = ("all configurations" if args.detsign_every == 1 or bridge
                 else f"every {args.detsign_every}th configuration")
        print(f"  determinant sign: {nsign[1]} x +1, {nsign[-1]} x -1 ({which})")
        if bridge:
            print("  S_fermion_phi: fermion action at the unshifted field (the HMC's S holds it at phi + c_k)")
        else:
            print(f"  S_fermion_phi vs S - S_gauge of the HMC: max deviation {sf_dev:.1e}")
    if bridge:
        with h5py.File(out, "r") as f:
            lam = f["config/bridge_lambda"][()]
        w = 1 / lam
        print(f"  bridge link: 1/Lambda in [{w.min():.3g}, {w.max():.3g}] (bound {1 / np.cos(bridge[0] / 2) ** 2:.3g}), "
              f"mean {w.mean():.4f}, N_eff/N = {w.sum() ** 2 / (w ** 2).sum() / len(w):.3f}; max |Re n - 1/2| / max(1, |n|) = {re_n_dev:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
