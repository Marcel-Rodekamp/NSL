#!/usr/bin/env python3
"""Stage-2 reduction: two-body correlators (excitons, pp pairs) of a measured PPP (or Hubbard) ensemble.

    python ppp_reduce2.py chain.yml [-o reduced2.h5] [--h5 chain.h5] [--window-energy 1.0] [--bin 1]
                          [--single] [--max-configs N] [--check 2] [--overwrite]

Input as for ppp_reduce.py (stage 1): the yml file and the h5 file with position-space propagators
(Executables/Hubbard/propagator, type: spatial); only the particle species is read. The orbital basis is the
one of stage 1 (symmetry adapted, valence orbitals = sublattice images of the conduction orbitals).

Operators, built from the orbitals of the window (conduction orbitals with 0 < eps <= window energy, and
their valence partners), grouped by symmetry sector (the product of the symmetry classes of the two orbitals):
  exciton     O^+_i = c^+_{c_i} c_{v_i} (c conduction, v valence), spin singlet or triplet
  pp singlet  P^+_i = (c^+_{a,up} c^+_{b,down} + c^+_{b,up} c^+_{a,down}) / sqrt(2), a <= b conduction (a = b: c^+_{a,up} c^+_{a,down})
  pp triplet  the same with a minus sign, a < b (S_z = 0 member)
hh pairs are not needed: on every configuration (mu = 0, bipartite) they are the complex conjugates of the pp
correlators of the partner orbitals, as R and A in stage 1. Combinations of definite particle-hole parity,
(c, v) +- (partner(v), partner(c)), are formed in the analysis from the pair basis stored here.

Per configuration (contractions in ppp_wick.py, tested against exact Fock-space traces):
  connected     X_ij(t) = -Fo(c_i, c_j) Bo(v_j, v_i), averaged over the source times: the triplet correlator
  disconnected  D_ij(t) = 1/Nt sum_t0 <O_i(t0 + t)> <O^+_j(t0)>, averaged over ALL time slices t0 (from G(t, t)):
                the singlet correlator is X + 2 D; for operators with the quantum numbers of the vacuum subtract
                2 <O_i><O^+_j> (ensemble averages of `expectation`) in the analysis
  expectation   time averages of <O_i> and <O^+_i> (shape (2, n))
  pp            sum over sources of Tr[B_i^+ Fo B_j Fo^T] / #sources
Products of two propagators are periodic in time, so no antiperiodic signs enter. The t = 0 entries of the exciton
correlators are not the operator-ordered equal-time values (the backward propagator at equal time is taken as
<c c^+> instead of -<c^+ c>): use t >= 1 (spectroscopy does anyway).

Output (plain h5py; b = bin of `--bin` consecutive configurations):
  attrs                                 as in stage 1, plus stage = 2, window_energy, bin, n_bins
  orbitals/{U, eps, class, class_names, partner, window}
  config/first_id                       first configuration id of every bin
  exciton/<sector>/pairs                (n, 2) orbital indices (c, v)
  exciton/<sector>/{connected, disconnected}   (b, Nt, n, n)
  exciton/<sector>/expectation          (b, 2, n)
  pp_singlet/<sector>/{pairs, correlator}, pp_triplet/<sector>/{pairs, correlator}   (n, 2), (b, Nt, n, n)
  checks/attrs                          max deviation of the pair-basis contractions from the general ones
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np
import h5py
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppp_ed  # noqa: E402
import ppp_wick as wick  # noqa: E402
import ppp_reduce as r1  # noqa: E402

VERSION = 1


def sector_name(names, ca, cb):
    """product of the characters of two symmetry classes ('+-' x '--' = '-+'); 'all' without symmetry"""
    a, b = names[ca], names[cb]
    if a == "all":
        return "all"
    return "".join("+" if x == y else "-" for x, y in zip(a, b))


def operator_bases(orb, window_energy, tol=1e-8):
    eps, cls, partner, names = orb["eps"], orb["cls"], orb["partner"], orb["class_names"]
    cond = [int(o) for o in np.argsort(eps) if 0 < eps[o] <= window_energy + tol]
    if not cond:
        raise SystemExit(f"no conduction orbital with 0 < eps <= {window_energy}")
    if any(partner[o] < 0 for o in cond):
        raise SystemExit("the window contains orbitals without a valence partner (not bipartite, or zero modes)")
    val = [int(partner[o]) for o in cond]
    exc, pps, ppt = {}, {}, {}
    for c in cond:
        for v in val:
            exc.setdefault(sector_name(names, cls[c], cls[v]), []).append((c, v))
    for i, a in enumerate(cond):
        for b in cond[i:]:
            s = sector_name(names, cls[a], cls[b])
            pps.setdefault(s, []).append((a, b))
            if a != b:
                ppt.setdefault(s, []).append((a, b))
    window = np.array(sorted(cond + val, key=lambda o: eps[o]))
    return window, {k: np.array(v) for k, v in exc.items()}, {k: np.array(v) for k, v in pps.items()}, \
        {k: np.array(v) for k, v in ppt.items()}


def cross_correlation(o, ob):
    """D[t, i, j] = 1/Nt sum_t0 o[(t0 + t) % Nt, i] ob[t0, j] for complex o, ob (circular cross-correlation:
    DFT of D = DFT(o)_k * DFT(ob)_{-k} / Nt = fft(o) * ifft(ob))"""
    return np.fft.ifft(np.fft.fft(o, axis=0)[:, :, None] * np.fft.ifft(ob, axis=0)[:, None, :], axis=0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("yml")
    p.add_argument("-o", "--output", help="default: reduced2/<yml name>.h5")
    p.add_argument("--h5", help="ensemble file (default: fileIO: h5file of the yml)")
    p.add_argument("--window-energy", type=float, default=1.0,
                   help="conduction orbitals with 0 < eps <= this (in t) and their partners [1.0]")
    p.add_argument("--bin", type=int, default=1, help="average this many consecutive configurations [1]")
    p.add_argument("--single", action="store_true", help="store the correlators in single precision")
    p.add_argument("--max-configs", type=int, default=0, help="reduce only the first N configurations (tests)")
    p.add_argument("--check", type=int, default=2, help="configurations checked against the general contractions")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    with open(args.yml) as fy:
        yml_text = fy.read()
    yml = yaml.safe_load(yml_text)
    sysyml = yml["system"]
    name, Nt, beta = sysyml["name"], int(sysyml["Nt"]), float(sysyml["beta"])
    Nx, mu = int(sysyml["nions"]), float(sysyml.get("mu", 0.0))
    delta = beta / Nt
    h5in = args.h5 or yml["fileIO"]["h5file"]
    out = args.output or os.path.join("reduced2", os.path.splitext(os.path.basename(args.yml))[0] + ".h5")
    if os.path.exists(out) and not args.overwrite:
        print(f"{out} exists; skipped (use --overwrite to redo)")
        return 0
    if abs(mu) > 1e-12:
        raise SystemExit("stage 2 assumes mu = 0 (particle-hole partners of the orbitals)")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    K = ppp_ed.hopping_matrix(sysyml)
    V = r1.interaction(yml, Nx)
    pos = np.array(sysyml["positions"], float)
    orb = r1.orbital_basis(K, V, pos, mu)
    U = orb["U"]
    window, exc, pps, ppt = operator_bases(orb, args.window_energy)
    wpos = {int(o): k for k, o in enumerate(window)}                 # orbital index -> position in the window
    Uw = U[:, window]
    loc = lambda arr: np.array([wpos[int(o)] for o in arr])          # noqa: E731
    ctype = np.complex64 if args.single else np.complex128
    B = max(1, args.bin)

    t_start = time.time()
    with h5py.File(h5in, "r") as fin:
        chain = fin[f"{name}/markovChain"]
        ids = sorted(int(k) for k in chain.keys() if k.isdigit())
        ids = [i for i in ids if f"{i}/propagator/spatial/particle" in chain]
        if args.max_configs > 0:
            ids = ids[: args.max_configs]
        nb = len(ids) // B
        if nb == 0:
            raise SystemExit(f"fewer than {B} measured configurations in {h5in}")
        sources = r1.sources_of(chain[f"{ids[0]}/propagator/spatial/particle"])
        print(f"{h5in}: {name}, {len(ids)} configurations -> {nb} bins of {B}; window {len(window)} orbitals "
              f"(|eps| <= {args.window_energy}); exciton sectors "
              f"{', '.join(f'{k} ({len(v)})' for k, v in exc.items())}; pp singlet {sum(map(len, pps.values()))}, "
              f"triplet {sum(map(len, ppt.values()))}")

        tmp = out + ".tmp"
        with h5py.File(tmp, "w") as fo:
            fo.attrs.update(dict(name=name, Nt=Nt, beta=beta, delta=delta, Nx=Nx, mu=mu, stage=2, version=VERSION,
                                 sources=np.array(sources), window_energy=args.window_energy, bin=B, n_bins=nb,
                                 n_configs=nb * B, input_h5=os.path.abspath(h5in), input_yml=os.path.abspath(args.yml),
                                 input_yml_text=yml_text, precision="single" if args.single else "double",
                                 created=datetime.datetime.now().isoformat(timespec="seconds")))
            fo["orbitals/U"], fo["orbitals/eps"], fo["orbitals/class"] = U, orb["eps"], orb["cls"]
            fo["orbitals/partner"], fo["orbitals/window"] = orb["partner"], window
            fo["orbitals/class_names"] = np.array(orb["class_names"], dtype="S")
            fo["config/first_id"] = np.array(ids[: nb * B : B])

            dsets = {}
            for s, pairs in exc.items():
                n = len(pairs)
                g = fo.create_group(f"exciton/{s}")
                g["pairs"] = pairs
                dsets[("exc", s)] = (g.create_dataset("connected", (nb, Nt, n, n), ctype),
                                     g.create_dataset("disconnected", (nb, Nt, n, n), ctype),
                                     g.create_dataset("expectation", (nb, 2, n), complex))
            for kind, basis in (("pp_singlet", pps), ("pp_triplet", ppt)):
                for s, pairs in basis.items():
                    g = fo.create_group(f"{kind}/{s}")
                    g["pairs"] = pairs
                    dsets[(kind, s)] = g.create_dataset("correlator", (nb, Nt, len(pairs), len(pairs)), ctype)

            tt = np.arange(Nt)
            worst = {"exciton": 0.0, "pp": 0.0}
            acc = None
            for n_cfg, i in enumerate(ids[: nb * B]):
                gp = chain[f"{i}/propagator/spatial/particle"]
                res = {}
                for s_src in sources:
                    idx = (s_src + tt) % Nt
                    Fo = Uw.T @ r1.read_tensor(gp[f"invM[t,{s_src}]"])[idx] @ Uw     # U^T G(s+t, s) U
                    Bo = Uw.T @ r1.read_tensor(gp[f"invM[{s_src},t]"])[idx] @ Uw     # U^T G(s, s+t) U
                    for s, pairs in exc.items():
                        X = wick.exciton_pairs(Fo, Bo, loc(pairs[:, 0]), loc(pairs[:, 1]))
                        res[("exc", s)] = res.get(("exc", s), 0) + X / len(sources)
                    for kind, basis, sign in (("pp_singlet", pps, +1), ("pp_triplet", ppt, -1)):
                        for s, pairs in basis.items():
                            P = wick.pp_pairs(Fo, loc(pairs[:, 0]), loc(pairs[:, 1]), sign)
                            res[(kind, s)] = res.get((kind, s), 0) + P / len(sources)
                Gtt = r1.read_tensor(gp["invM[t,t]"])
                Eo = np.eye(len(window)) - np.swapaxes(Uw.T @ Gtt @ Uw, 1, 2)          # <c^+ c> per spin
                for s, pairs in exc.items():
                    o, ob = wick.exciton_expectations(Eo, loc(pairs[:, 0]), loc(pairs[:, 1]))
                    res[("dis", s)] = cross_correlation(o, ob)
                    res[("ex", s)] = np.stack([o.mean(axis=0), ob.mean(axis=0)])

                if n_cfg < args.check:                                                  # general contractions, site basis
                    s0 = sources[0]
                    col, row = r1.read_tensor(gp[f"invM[t,{s0}]"]), r1.read_tensor(gp[f"invM[{s0},t]"])
                    for t in (1, Nt // 4, Nt // 2):
                        tg = (s0 + t) % Nt
                        Gf, Gb = col[tg], row[tg]
                        s, pairs = next(iter(exc.items()))
                        c0, v0 = pairs[0]
                        A = np.outer(U[:, c0], U[:, v0])
                        Xg, Sg = wick.exciton(A, A, Gf, Gb, Gtt[tg], Gtt[s0])
                        Fo1 = Uw.T @ Gf @ Uw
                        Bo1 = Uw.T @ Gb @ Uw
                        Xf = wick.exciton_pairs(Fo1, Bo1, loc([c0]), loc([v0]))[0, 0]
                        o1, _ = wick.exciton_expectations(Eo[tg], loc([c0]), loc([v0]))
                        _, ob1 = wick.exciton_expectations(Eo[s0], loc([c0]), loc([v0]))
                        Sf = Xf + 2 * o1[0] * ob1[0]
                        worst["exciton"] = max(worst["exciton"], abs(Xg - Xf) / abs(Xg), abs(Sg - Sf) / abs(Sg))
                        a0, b0 = next(iter(pps.values()))[-1]
                        Bm = wick.pair_matrix(U[:, a0], U[:, b0], +1)
                        Pg = wick.pp(Bm, Bm, Gf)
                        Pf = wick.pp_pairs(Fo1, loc([a0]), loc([b0]), +1)[0, 0]
                        worst["pp"] = max(worst["pp"], abs(Pg - Pf) / abs(Pg))

                if acc is None:
                    acc = {k: np.zeros_like(v) for k, v in res.items()}
                for k, v in res.items():
                    acc[k] += v / B
                if (n_cfg + 1) % B == 0:
                    b = n_cfg // B
                    for s in exc:
                        dsets[("exc", s)][0][b] = acc[("exc", s)]
                        dsets[("exc", s)][1][b] = acc[("dis", s)]
                        dsets[("exc", s)][2][b] = acc[("ex", s)]
                    for kind, basis in (("pp_singlet", pps), ("pp_triplet", ppt)):
                        for s in basis:
                            dsets[(kind, s)][b] = acc[(kind, s)]
                    acc = None
                    if b % max(1, nb // 10) == 0 or b == nb - 1:
                        print(f"  bin {b + 1}/{nb}, {time.time() - t_start:.0f} s", flush=True)
            chk = fo.create_group("checks")
            chk.attrs["exciton_max_rel_dev"] = worst["exciton"]
            chk.attrs["pp_max_rel_dev"] = worst["pp"]
            chk.attrs["checked_configs"] = min(args.check, len(ids))
    os.replace(tmp, out)
    print(f"wrote {out} ({os.path.getsize(out) / 2**20:.0f} MB) in {time.time() - t_start:.0f} s")
    print(f"  check against the general contractions (first {min(args.check, len(ids))} configs): "
          f"exciton {worst['exciton']:.1e}, pp {worst['pp']:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
