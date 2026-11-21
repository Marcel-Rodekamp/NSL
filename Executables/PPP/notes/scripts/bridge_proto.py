"""Python prototype of the bridge link (deck 7): finite temperature, charge-basis HS, half-filled Hubbard on a
4-site open chain (U = 4, beta = 4, Nt = 16). Independent of NSL; it produced the chains behind the figures.

Model: B_t = e^{delta K} diag(e^{i phi_t}), M = 1 - (antiperiodic subdiagonal B_t); weight e^{-S_g} |det M|^2,
S_g = sum phi^2 / (2 delta U). Exact reference: Fock space, W = e^{delta K^} e^{-delta U/2 sum q^2}.

Bridge: Lambda = 1/(2 Nx) sum_{i, x=+-1} e^{i theta x q_i} at slice b, i.e. the fermion weight at the field shifted by
c_{i,x} = theta x e_{b,i}. Joint chain on (phi, k = (i, x)): HMC on phi with |det M(phi + c_k)|^2, heat bath on k.
Marginal in phi: e^{-S_g} |det M(phi)|^2 Lam(phi), Lam = mean_k |1 + (e^{i theta x} - 1) n_i(b)|^2, n = 1 - diag G(b, b).
Estimator: <O> = <O / Lam> / <1 / Lam>.

    python bridge_proto.py check                                   # the identities used by the bridge
    python bridge_proto.py run {plain,bridge} NTRAJ SEED OUT.npz   # one chain (100000 trajectories: about 4 min)
    python bridge_proto.py exact OUT.npy                           # Fock-space S_AF, S_CDW, <s^2>

OUT.npz holds per trajectory: obs (S_AF, S_CDW, <s^2>), lam (Lambda, also for plain chains), sign (of
e^{-i sum phi/2} det M), acc, k (bridge term). Figures: bridge_figures.py.
"""
import os
import sys
import time

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import ppp_wick as wick  # noqa: E402

Nx, U, beta, Nt = 4, 4.0, 4.0, 16
delta = beta / Nt
K = np.zeros((Nx, Nx))
for i in range(Nx - 1):
    K[i, i + 1] = K[i + 1, i] = 1.0
eK = sla.expm(delta * K)
sub = (-1.0) ** np.arange(Nx)
THETA, BSLICE = np.pi / 2, 0


def fermion_matrix(phi):
    M = np.eye(Nt * Nx, dtype=complex)
    for t in range(Nt):
        B = eK * np.exp(1j * phi[t])[None, :]
        r = (t + 1) % Nt
        M[r * Nx:(r + 1) * Nx, t * Nx:(t + 1) * Nx] = -B if t < Nt - 1 else B
    return M


def green(phi):
    M = fermion_matrix(phi)
    G = np.linalg.inv(M)
    Gt = np.array([G[t * Nx:(t + 1) * Nx, t * Nx:(t + 1) * Nx] for t in range(Nt)])   # G(t,t) = <c c^+>
    return M, Gt


def density(Gt):
    return 1.0 - np.diagonal(Gt, axis1=-2, axis2=-1)          # n_x(t), generalized, per spin


def shifts():
    return [(i, x) for i in range(Nx) for x in (+1, -1)]


def bridge_ratios(n_b):
    """R_k = |det M(phi + c_k)|^2 / |det M(phi)|^2 for all k, from n at the bridge slice"""
    return np.array([abs(1 + (np.exp(1j * THETA * x) - 1) * n_b[i]) ** 2 for i, x in shifts()])


def observables(Gt):
    """equal-time observables quadratic in G, averaged over t: S_AF, S_CDW, <s_x^2> mean"""
    _, ss, qq = wick.equal_time(Gt)
    ss, qq = ss.mean(axis=0), qq.mean(axis=0)
    return np.array([np.real(sub @ ss @ sub) / Nx, np.real(sub @ qq @ sub) / Nx, np.real(np.trace(ss)) / Nx])


def check():
    rng = np.random.default_rng(1)
    worst = dict(ratio=0.0, re_half=0.0, real_d=0.0, bound=np.inf)
    for _ in range(50):
        phi = rng.normal(0, np.sqrt(delta * U), (Nt, Nx))
        M, Gt = green(phi)
        sd, ld = np.linalg.slogdet(M)
        d = sd * np.exp(-0.5j * phi.sum())
        worst["real_d"] = max(worst["real_d"], abs(d.imag) / abs(d))
        n = density(Gt)
        worst["re_half"] = max(worst["re_half"], np.abs(n.real - 0.5).max())
        for (i, x), R in zip(shifts(), bridge_ratios(n[BSLICE])):
            ph = phi.copy(); ph[BSLICE, i] += THETA * x
            s2, l2 = np.linalg.slogdet(fermion_matrix(ph))
            worst["ratio"] = max(worst["ratio"], abs(np.exp(2 * (l2 - ld)) - R) / R)
        worst["bound"] = min(worst["bound"], bridge_ratios(n[BSLICE]).mean() / np.cos(THETA / 2) ** 2)
    print(f"e^(-i sum phi/2) det M real: max |Im|/|d| {worst['real_d']:.1e}")
    print(f"Re n_x(t) = 1/2 on every configuration: max dev {worst['re_half']:.1e}")
    print(f"rank-1 ratio |det M(phi + c_k)|^2/|det M(phi)|^2 = |1 + (e^(i theta x)-1) n_i(b)|^2: max rel dev {worst['ratio']:.1e}")
    print(f"Lam / cos^2(theta/2) >= 1: min {worst['bound']:.4f}")


def force(phi):
    """dS/dphi with S = S_g - 2 log|det M(phi)|; d log det / d phi_{t,x} = i n_x(t)"""
    _, Gt = green(phi)
    return phi / (delta * U) + 2 * np.imag(density(Gt))


def action(phi):
    return 0.5 * np.sum(phi ** 2) / (delta * U) - 2 * np.linalg.slogdet(fermion_matrix(phi))[1]


def hmc(phi, c, rng, nmd=6, T=np.pi / 2):
    """kinetic p^2 delta U / 2, so the Gaussian part rotates with frequency 1 (exact), fermion force as kicks"""
    m = 1.0 / (delta * U)
    p = rng.normal(0, np.sqrt(m), phi.shape)
    H0 = 0.5 * np.sum(p ** 2) / m + action(phi + c) - 0.5 * np.sum((phi + c) ** 2) / (delta * U) + 0.5 * np.sum(phi ** 2) / (delta * U)
    h = T / nmd
    x, q = phi.copy(), p.copy()

    def ferm_force(y):
        _, Gt = green(y + c)
        return 2 * np.imag(density(Gt))

    q -= 0.5 * h * ferm_force(x)
    for s in range(nmd):
        cs, sn = np.cos(h), np.sin(h)                       # exact harmonic rotation, omega = 1
        x, q = cs * x + sn * q / m, -sn * m * x + cs * q
        q -= (h if s < nmd - 1 else 0.5 * h) * ferm_force(x)
    H1 = 0.5 * np.sum(q ** 2) / m + 0.5 * np.sum(x ** 2) / (delta * U) - 2 * np.linalg.slogdet(fermion_matrix(x + c))[1]
    dH = H1 - (0.5 * np.sum(p ** 2) / m + 0.5 * np.sum(phi ** 2) / (delta * U)
               - 2 * np.linalg.slogdet(fermion_matrix(phi + c))[1])
    acc = np.isfinite(dH) and rng.random() < np.exp(min(0.0, -dH))
    return (x if acc else phi), acc


def run(kind, ntraj, seed, out):
    rng = np.random.default_rng(seed)
    phi = np.zeros((Nt, Nx))
    k = 0
    zero = np.zeros((Nt, Nx))
    obs, lam, sgn, accs, ks = [], [], [], [], []
    t0 = time.time()
    for n in range(ntraj):
        c = zero
        if kind == "bridge":
            i, x = shifts()[k]
            c = zero.copy(); c[BSLICE, i] = THETA * x
        phi, a = hmc(phi, c, rng)
        M, Gt = green(phi)
        nb = density(Gt)[BSLICE]
        R = bridge_ratios(nb)
        if kind == "bridge":                                # heat bath on the bridge term given phi
            k = int(rng.choice(len(R), p=R / R.sum()))
        obs.append(observables(Gt)); lam.append(R.mean()); accs.append(a); ks.append(k)
        sd, _ = np.linalg.slogdet(M)
        sgn.append(np.sign((sd * np.exp(-0.5j * phi.sum())).real))
        if (n + 1) % 2000 == 0:
            print(f"{kind} {n + 1}/{ntraj} {time.time() - t0:.0f} s acc {np.mean(accs):.2f}", flush=True)
    np.savez(out, obs=np.array(obs), lam=np.array(lam), sign=np.array(sgn), acc=np.array(accs), k=np.array(ks))


def exact(out):
    m = 2 * Nx
    sm, Z2, I2 = np.array([[0, 1], [0, 0]]), np.diag([1, -1]), np.eye(2)

    def ann(j):
        ops = [Z2] * j + [sm] + [I2] * (m - j - 1)
        out = ops[0]
        for o in ops[1:]:
            out = np.kron(out, o)
        return out.astype(float)
    c = [ann(j) for j in range(m)]
    n = [x.T @ x for x in c]
    Kh = sum(K[x, y] * (c[x].T @ c[y] + c[x + Nx].T @ c[y + Nx]) for x in range(Nx) for y in range(Nx) if K[x, y])
    q = [n[x] + n[x + Nx] - np.eye(2 ** m) for x in range(Nx)]
    s = [n[x] - n[x + Nx] for x in range(Nx)]
    HV = 0.5 * U * sum(qx @ qx for qx in q)
    W = sla.expm(delta * Kh) @ sla.expm(-delta * HV)
    WN = np.linalg.matrix_power(W, Nt)
    Z = np.trace(WN)
    ev = lambda O: np.trace(WN @ O) / Z                       # noqa: E731
    saf = sum(sub[x] * sub[y] * ev(s[x] @ s[y]) for x in range(Nx) for y in range(Nx)) / Nx
    scdw = sum(sub[x] * sub[y] * ev(q[x] @ q[y]) for x in range(Nx) for y in range(Nx)) / Nx
    s2 = sum(ev(s[x] @ s[x]) for x in range(Nx)) / Nx
    print("exact S_AF, S_CDW, <s^2>:", np.round([saf, scdw, s2], 6))
    np.save(out, np.real(np.array([saf, scdw, s2])))


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    elif sys.argv[1] == "exact":
        exact(sys.argv[2])
    else:
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
