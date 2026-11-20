#!/usr/bin/env python3
"""Wick contractions of NSL's charge-basis propagators, one configuration at a time.

For a fixed auxiliary field the fermions are free, so every correlator of a configuration is a
sum of products of its propagators (Wick's theorem); the physical correlator is the ensemble
average of these per-configuration values. All formulas are checked against exact Fock-space
traces in test_ppp_wick.py.

Conventions (HubbardExp, charge basis):
  G = M_p^{-1}, the particle (spin-up electron) propagator of one configuration,
      G[t, x, t0, y] = <c_x(t) c^+_y(t0)>  for t > t0,   G[t, x, t, y] = <c_x c^+_y>  (equal time).
  The auxiliary field couples to the charge, so both spins see the same field: the spin-down
  propagator equals G. NSL's hole species (h_x = c^+_{x,down}) is redundant:
      G_h(t0, t)^T = -G(t, t0)  (t > t0),      G_h(t, t)^T = 1 - G(t, t).
  Below Gf = G(t, t0) and Gb = G(t0, t) are (..., Nx, Nx) matrices for t > t0 (leading axes,
  e.g. the time t, broadcast), Gt = G(t, t) and Gt0 = G(t0, t0).

Equal times (t = t0, tau = 0): the product keeps its operator order, sink left of source. A sink c
contracted with a source c^+ gives <c c^+> = G(t0, t0), the t -> t0 limit of the forward propagator, as
stored in M^-1[t0, t0]. A sink c^+ contracted with a source c gives -<c^+ c>^T = G(t0, t0) - 1, the limit of
the backward propagator (backward_equal_time). So at t = t0 pass Gf = G(t0, t0) and Gb = G(t0, t0) - 1.
The extra -1 per backward propagator is the contact term: the exciton gains + Tr[A_i^+ G(t0, t0) A_j],
hh gains a delta-delta term and two delta-G terms, and pp has none (forward propagators only). With M^-1
for a backward propagator at t = t0, the correlator is not <O O^+> at tau = 0.

Operators (A, B are Nx x Nx coefficient matrices, e.g. outer products of hopping orbitals):
  exciton    O^+ = sum_xy A_xy c^+_{x,s} c_{y,s'}      (electron into A's row space, out of its column space)
             triplet: s != s' (S_z = +-1), or the S_z = 0 combination of s = s' (identical per configuration)
             singlet: (c^+_up A c_up + c^+_down A c_down) / sqrt(2)
  pp         P^+ = sum_xy B_xy c^+_{x,up} c^+_{y,down}  singlet for symmetric B, S_z = 0 triplet for antisymmetric B
  hh         H^+ = sum_xy B_xy c_{x,up} c_{y,down}     removes two electrons; same spin structure
The correlators are <O(t) O^+(t0)> with O = (O^+)^+, for a sink operator A_i and a source A_j.
"""

import numpy as np


def _dag(A):
    return np.conj(np.swapaxes(A, -1, -2))


def density_matrix(Gt):
    """E[x, y] = <c^+_x c_y> at equal time, per spin, from Gt = G(t, t)"""
    return np.eye(Gt.shape[-1]) - np.swapaxes(Gt, -1, -2)


def backward_equal_time(Gt0):
    """the backward propagator G(t0, t) for t -> t0 from above: -<c^+_y c_x> = G(t0, t0) - 1 (see the module doc).
    Use it as Gb (or, in an orthonormal orbital basis U, Bo = U^T G U - 1) at t = t0."""
    return Gt0 - np.eye(Gt0.shape[-1])


def equal_time(Gt):
    """Equal-time two-body observables of one configuration from Gt = G(t, t), shape (..., Nx, Nx).

    Returns (E, ss, qq): E = <c^+_x c_y> per spin, ss = <s_x s_y> with s = n_up - n_down and
    qq = <q_x q_y> with q = n_up + n_down - 1. <S_x . S_y> = 3/4 ss (spin symmetry per configuration)."""
    E = density_matrix(Gt)
    exch = 2.0 * E * Gt                                   # 2 <c^+_x c_y><c_x c^+_y>, summed over the spin
    q = 2.0 * np.diagonal(E, axis1=-2, axis2=-1) - 1.0    # <q_x>
    return E, exch, q[..., :, None] * q[..., None, :] + exch


def exciton(A_sink, A_src, Gf, Gb, Gt=None, Gt0=None):
    """Exciton correlators <O_i(t) O^+_j(t0)>; returns (triplet, singlet).

    triplet = -Tr[A_i^+ Gf A_j Gb] (no disconnected part); singlet = triplet + 2 <O_i(t)> <O^+_j(t0)>.
    The singlet needs the equal-time Gt, Gt0; without them only the triplet is returned (singlet None).
    At t = t0: Gf = Gt0, Gb = backward_equal_time(Gt0)."""
    X = -np.einsum("xy,...xa,ab,...by->...", np.conj(A_sink), Gf, A_src, Gb)
    if Gt is None or Gt0 is None:
        return X, None
    Ot = np.einsum("xy,...yx->...", np.conj(A_sink), density_matrix(Gt))   # <O_i(t)>
    Od = np.einsum("xy,...xy->...", A_src, density_matrix(Gt0))            # <O^+_j(t0)>
    return X, X + 2.0 * Ot * Od


def pp(B_sink, B_src, Gf):
    """Two added electrons: <P_i(t) P^+_j(t0)> = Tr[B_i^+ Gf B_j Gf^T]"""
    return np.einsum("xy,...xa,ab,...yb->...", np.conj(B_sink), Gf, B_src, Gf)


def hh(B_sink, B_src, Gb):
    """Two removed electrons: <H_i(t) H^+_j(t0)> = Tr[B_i^+ Gb^T B_j Gb] with Gb = G(t0, t)
    (at t = t0: Gb = backward_equal_time(Gt0))"""
    return np.einsum("xy,...ax,ab,...by->...", np.conj(B_sink), Gb, B_src, Gb)


# --- the same contractions for operator bases of orbital pairs (fast; checked against the general ones) ----
# Fo = U^T G(t, t0) U, Bo = U^T G(t0, t) U, Eo = U^T E(t) U in the orbital basis U (real orthogonal); c, v, a, b are
# integer arrays of orbital indices, one entry per operator of the basis.

def exciton_pairs(Fo, Bo, c, v):
    """connected exciton correlator matrix of O^+_i = c^+_{c_i} c_{v_i} (spin as in `exciton`):
    X[..., i, j] = -Fo[..., c_i, c_j] Bo[..., v_j, v_i] = triplet (any S_z); singlet = X + 2 <O_i(t)><O^+_j(t0)>.
    At t = t0: Bo = U^T G(t0, t0) U - 1, which adds the contact term delta_{v_i v_j} Fo[c_i, c_j]."""
    c, v = np.asarray(c), np.asarray(v)
    return -Fo[..., c[:, None], c[None, :]] * Bo[..., v[None, :], v[:, None]]


def exciton_expectations(Eo, c, v):
    """(<O_i>, <O^+_i>) = (<c^+_{v_i} c_{c_i}>, <c^+_{c_i} c_{v_i}>) from the density matrix Eo(t) = <c^+ c> per spin"""
    return Eo[..., v, c], Eo[..., c, v]


def pp_pairs(Fo, a, b, sign):
    """pp correlator matrix of P^+_i = (c^+_{a_i,up} c^+_{b_i,down} + sign c^+_{b_i,up} c^+_{a_i,down}) / sqrt(2)
    (a_i != b_i), or c^+_{a,up} c^+_{a,down} for a_i == b_i (sign = +1): singlet for sign = +1, S_z = 0 triplet
    for sign = -1. P_ij = 2 al_i al_j [Fo(a_i,a_j) Fo(b_i,b_j) + sign Fo(a_i,b_j) Fo(b_i,a_j)], al = 1/sqrt(2) or 1/2."""
    a, b = np.asarray(a), np.asarray(b)
    al = np.where(a == b, 0.5, 1 / np.sqrt(2.0))
    F = lambda p, q: Fo[..., p[:, None], q[None, :]]           # noqa: E731
    return 2 * al[:, None] * al[None, :] * (F(a, a) * F(b, b) + sign * F(a, b) * F(b, a))


def pair_matrix(u, v, sign=+1):
    """(u v^T + sign v u^T)/sqrt(2) for u != v, u u^T for u == v: B of a pp/hh pair of orbitals"""
    if u is v or np.allclose(u, v):
        return np.outer(u, u)
    return (np.outer(u, v) + sign * np.outer(v, u)) / np.sqrt(2.0)
