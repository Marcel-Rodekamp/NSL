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
Products of two propagators are periodic in time, so no antiperiodic signs enter. At t = 0 the backward
propagator is G(s, s) - 1 = -<c^+ c>^T (ppp_wick.backward_equal_time), not M^-1[s, s] = <c c^+>: this contact
term, + delta_{v_i v_j} G(s, s)_{c_i c_j} in X, makes every stored t = 0 entry the operator-ordered equal-time
value <O_i O^+_j> (attribute contact_terms = 1). D and the pp correlators need none: D is built from equal-time
expectation values, and pp uses forward propagators only. Files of VERSION 1 lack the term in X(0); ppp_analysis
adds it from the stage-1 file of the same chain.

Bridge link (chains generated with `HMC: bridge:`): as in stage 1, every per-configuration quantity is multiplied
by the weight 1/Lambda before binning, and `weights` holds the bin averages of the weight (all 1 without the bridge
link); ppp_analysis forms the ratios.

Output (plain h5py; b = bin of `--bin` consecutive configurations):
  attrs                                 as in stage 1, plus stage = 2, window_energy, bin, n_bins, contact_terms,
                                        bridge, bridge_theta, bridge_slice
  weights                               (b,) bin averages of 1/Lambda; the datasets below are bin averages of weight * O
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

VERSION = 3                     # 2: contact term in the exciton X(0); 3: bridge-link weights


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


def static_setup(yml, window_energy):
    """Orbital basis (as stage 1), window and operator bases of a run: everything of stage 2 that does not depend on
    the configurations"""
    sysyml = yml["system"]
    Nt, beta = int(sysyml["Nt"]), float(sysyml["beta"])
    Nx, mu = int(sysyml["nions"]), float(sysyml.get("mu", 0.0))
    if abs(mu) > 1e-12:
        raise SystemExit("stage 2 assumes mu = 0 (particle-hole partners of the orbitals)")
    K = ppp_ed.hopping_matrix(sysyml)
    V = r1.interaction(yml, Nx)
    pos = np.array(sysyml["positions"], float)
    orb = r1.orbital_basis(K, V, pos, mu)
    window, exc, pps, ppt = operator_bases(orb, window_energy)
    return dict(name=sysyml["name"], Nt=Nt, beta=beta, delta=beta / Nt, Nx=Nx, mu=mu, offset=float(sysyml.get("offset", 0.0)),
                orb=orb, window=window, exc=exc, pps=pps, ppt=ppt)


def write_static(fo, st, **attrs):
    """The attributes (those of static_setup and `attrs`), orbitals/ and the operator pairs of a stage-2 file"""
    orb = st["orb"]
    fo.attrs.update(dict(name=st["name"], Nt=st["Nt"], beta=st["beta"], delta=st["delta"], Nx=st["Nx"], mu=st["mu"],
                         stage=2, version=VERSION, contact_terms=1, **attrs))
    fo["orbitals/U"], fo["orbitals/eps"], fo["orbitals/class"] = orb["U"], orb["eps"], orb["cls"]
    fo["orbitals/partner"], fo["orbitals/window"] = orb["partner"], st["window"]
    fo["orbitals/class_names"] = np.array(orb["class_names"], dtype="S")
    for s, pairs in st["exc"].items():
        fo[f"exciton/{s}/pairs"] = pairs
    for kind, basis in (("pp_singlet", st["pps"]), ("pp_triplet", st["ppt"])):
        for s, pairs in basis.items():
            fo[f"{kind}/{s}/pairs"] = pairs


def check_pairs(st):
    """the orbital pairs (c, v) and (a, b) of the check against the general contractions (main: first configurations)"""
    c0, v0 = next(iter(st["exc"].values()))[0]
    a0, b0 = next(iter(st["pps"].values()))[-1]
    return (int(c0), int(v0)), (int(a0), int(b0))


def write_skeleton(yml_text, yml, out, B=1, single=False, window_energy=1.0, h5in=None, yml_path=None):
    """An empty stage-2 file for the measurement on the fly (src/NSL/Measurements/Hubbard/reducedObservables.tpp):
    the attributes, orbitals/ and operator pairs of main(), and every per-bin dataset with length 0, extendible along
    its first axis (n_configs = n_bins = 0). checks/attrs also hold the orbital pairs of the check against the general
    contractions (exciton_check_pair, pp_check_pair), which the C++ repeats on the first configurations."""
    st = static_setup(yml, window_energy)
    Nt = st["Nt"]
    meas = yml.get("measurements") or {}
    sources = r1.source_times(Nt, int(meas.get("Number Time Sources", 1)))
    bridge = r1.bridge_settings(yml, None, warn=False)
    if bridge is not None and abs(st["offset"]) > 1e-12:
        raise SystemExit("the bridge link needs offset = 0")
    ctype = np.complex64 if single else np.complex128
    h5in = h5in or yml["fileIO"]["h5file"]
    tmp = out + ".tmp"
    with h5py.File(tmp, "w") as fo:
        write_static(fo, st, sources=np.array(sources), window_energy=window_energy, bin=B, n_bins=0,
                     bridge=int(bridge is not None), bridge_theta=bridge[0] if bridge else np.nan,
                     bridge_slice=bridge[1] if bridge else -1, n_configs=0, input_h5=os.path.abspath(h5in),
                     input_yml=os.path.abspath(yml_path) if yml_path else "", input_yml_text=yml_text,
                     precision="single" if single else "double", on_the_fly=1,
                     created=datetime.datetime.now().isoformat(timespec="seconds"))

        def grow(path, shape, dtype, chunk=1):          # length 0, extendible along the first axis
            return fo.create_dataset(path, (0,) + tuple(shape), dtype, maxshape=(None,) + tuple(shape),
                                     chunks=(chunk,) + tuple(shape))
        grow("config/first_id", (), np.int64, 256)
        grow("weights", (), float, 256)
        for s, pairs in st["exc"].items():
            n = len(pairs)
            grow(f"exciton/{s}/connected", (Nt, n, n), ctype)
            grow(f"exciton/{s}/disconnected", (Nt, n, n), ctype)
            grow(f"exciton/{s}/expectation", (2, n), complex, 256)
        for kind, basis in (("pp_singlet", st["pps"]), ("pp_triplet", st["ppt"])):
            for s, pairs in basis.items():
                grow(f"{kind}/{s}/correlator", (Nt, len(pairs), len(pairs)), ctype)
        chk = fo.create_group("checks")
        chk.attrs["exciton_max_rel_dev"] = 0.0
        chk.attrs["pp_max_rel_dev"] = 0.0
        chk.attrs["checked_configs"] = 0
        (c0, v0), (a0, b0) = check_pairs(st)
        chk.attrs["exciton_check_pair"] = np.array([c0, v0])
        chk.attrs["pp_check_pair"] = np.array([a0, b0])
    os.replace(tmp, out)
    print(f"wrote the empty stage-2 file {out}: window {len(st['window'])} orbitals (|eps| <= {window_energy}), "
          f"exciton sectors {', '.join(f'{k} ({len(v)})' for k, v in st['exc'].items())}, bins of {B}")
    return 0


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
    p.add_argument("--skeleton", action="store_true",
                   help="write only the empty file for the measurement on the fly (write_skeleton); no ensemble needed")
    args = p.parse_args(argv)

    with open(args.yml) as fy:
        yml_text = fy.read()
    yml = yaml.safe_load(yml_text)
    sysyml = yml["system"]
    h5in = args.h5 or yml["fileIO"]["h5file"]
    out = args.output or os.path.join("reduced2", os.path.splitext(os.path.basename(args.yml))[0] + ".h5")
    if os.path.exists(out) and not args.overwrite:
        print(f"{out} exists; skipped (use --overwrite to redo)")
        return 0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if args.skeleton:
        return write_skeleton(yml_text, yml, out, max(1, args.bin), args.single, args.window_energy, h5in, args.yml)

    st = static_setup(yml, args.window_energy)
    name, Nt, beta, delta, Nx, mu = (st[k] for k in ("name", "Nt", "beta", "delta", "Nx", "mu"))
    orb, window, exc, pps, ppt = (st[k] for k in ("orb", "window", "exc", "pps", "ppt"))
    U = orb["U"]
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
        bridge = r1.bridge_settings(yml, fin.get(f"{name}/Meta/bridge"))
        if bridge is not None and abs(float(sysyml.get("offset", 0.0))) > 1e-12:
            raise SystemExit("the bridge link needs offset = 0")
        print(f"{h5in}: {name}, {len(ids)} configurations -> {nb} bins of {B}; window {len(window)} orbitals "
              f"(|eps| <= {args.window_energy}); exciton sectors "
              f"{', '.join(f'{k} ({len(v)})' for k, v in exc.items())}; pp singlet {sum(map(len, pps.values()))}, "
              f"triplet {sum(map(len, ppt.values()))}"
              + (f"; bridge link theta = {bridge[0]:.6g}, slice {bridge[1]}: weights 1/Lambda" if bridge else ""))

        tmp = out + ".tmp"
        with h5py.File(tmp, "w") as fo:
            write_static(fo, st, sources=np.array(sources), window_energy=args.window_energy, bin=B, n_bins=nb,
                         bridge=int(bridge is not None), bridge_theta=bridge[0] if bridge else np.nan,
                         bridge_slice=bridge[1] if bridge else -1, n_configs=nb * B, input_h5=os.path.abspath(h5in),
                         input_yml=os.path.abspath(args.yml), input_yml_text=yml_text,
                         precision="single" if args.single else "double",
                         created=datetime.datetime.now().isoformat(timespec="seconds"))
            fo["config/first_id"] = np.array(ids[: nb * B : B])
            dW = fo.create_dataset("weights", (nb,), float)
            wsum = 0.0

            dsets = {}
            for s, pairs in exc.items():
                n = len(pairs)
                g = fo[f"exciton/{s}"]
                dsets[("exc", s)] = (g.create_dataset("connected", (nb, Nt, n, n), ctype),
                                     g.create_dataset("disconnected", (nb, Nt, n, n), ctype),
                                     g.create_dataset("expectation", (nb, 2, n), complex))
            for kind, basis in (("pp_singlet", pps), ("pp_triplet", ppt)):
                for s, pairs in basis.items():
                    g = fo[f"{kind}/{s}"]
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
                    Bo[0] -= np.eye(len(window))                                    # t = 0: U^T (G(s,s) - 1) U
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
                    for t in (0, 1, Nt // 4, Nt // 2):
                        tg = (s0 + t) % Nt
                        Gf, Gb = col[tg], (row[tg] if t else wick.backward_equal_time(row[tg]))
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

                w = 1.0 / r1.bridge_lambda(Gtt[bridge[1]], bridge[0])[0] if bridge else 1.0
                if acc is None:
                    acc = {k: np.zeros_like(v) for k, v in res.items()}
                    wsum = 0.0
                for k, v in res.items():
                    acc[k] += w * v / B
                wsum += w
                if (n_cfg + 1) % B == 0:
                    b = n_cfg // B
                    for s in exc:
                        dsets[("exc", s)][0][b] = acc[("exc", s)]
                        dsets[("exc", s)][1][b] = acc[("dis", s)]
                        dsets[("exc", s)][2][b] = acc[("ex", s)]
                    for kind, basis in (("pp_singlet", pps), ("pp_triplet", ppt)):
                        for s in basis:
                            dsets[(kind, s)][b] = acc[(kind, s)]
                    dW[b] = wsum / B
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
