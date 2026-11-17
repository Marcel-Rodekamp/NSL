#!/usr/bin/env python3
"""Ideal benzenoid geometries and NSL yml files for PPP simulations.

Molecules are built from regular hexagons on a honeycomb lattice with unit bond
length (positions are in units of `length unit`, 1.4 Angstrom by default).
Hexagons are flat-topped (vertices at angles 60k deg); neighbouring hexagon
centres sit sqrt(3) apart in the directions 30+60k deg.

    python ppp_geometry.py perylene --beta 10 --Nt 40 -o perylene.yml

Hopping: every bond gets `--hopping` (units of t). Bonds that join two fused
units without belonging to a second ring (in perylene: the two inter-naphthalene
"bridge" bonds) can be given a different value via `--bridge-hopping`; they are
marked in the yml file so that they can also be edited by hand.
"""

import argparse
import math
import sys

import numpy as np

# hexagon centres (in units of the neighbour direction index k: angle 30+60k deg)
MOLECULES = {
    "benzene":     [()],
    "naphthalene": [(), (0,)],
    "anthracene":  [(), (0,), (0, 0)],
    "pyrene":      [(), (0,), (1,), (0, 1)],
    # central ring fused to two naphthalene units (rings at +-30 deg and 150/210 deg)
    "perylene":    [(), (0,), (5,), (2,), (3,)],
    "coronene":    [()] + [(k,) for k in range(6)],
}


def hexagon_centres(name):
    steps = MOLECULES[name]
    dirs = [np.sqrt(3) * np.array([np.cos(np.pi/6 + np.pi/3*k), np.sin(np.pi/6 + np.pi/3*k)]) for k in range(6)]
    return [sum((dirs[k] for k in path), np.zeros(2)) for path in steps]


def build(name):
    """Returns positions (N,2), bonds [(i,j)], rings [[site ids]] for a benzenoid."""
    centres = hexagon_centres(name)
    key = lambda v: (round(float(v[0]), 6), round(float(v[1]), 6))
    sites, index, rings = [], {}, []
    for c in centres:
        ring = []
        for k in range(6):
            v = c + np.array([np.cos(np.pi/3*k), np.sin(np.pi/3*k)])
            kv = key(v)
            if kv not in index:
                index[kv] = len(sites)
                sites.append(v)
            ring.append(index[kv])
        rings.append(ring)
    pos = np.array(sites)
    # canonical site order: by x, then y (keeps output stable)
    order = np.lexsort((np.round(pos[:, 1], 6), np.round(pos[:, 0], 6)))
    relabel = {int(old): new for new, old in enumerate(order)}
    pos = pos[order]
    rings = [[relabel[s] for s in r] for r in rings]
    bonds = sorted({tuple(sorted((r[k], r[(k+1) % 6]))) for r in rings for k in range(6)})
    return pos, bonds, rings


def bridge_bonds(bonds, rings, n):
    """Bonds between two 3-coordinated sites that belong to a single ring."""
    degree = np.zeros(n, int)
    for i, j in bonds:
        degree[i] += 1
        degree[j] += 1
    ring_count = {b: 0 for b in bonds}
    for r in rings:
        for k in range(6):
            ring_count[tuple(sorted((r[k], r[(k+1) % 6])))] += 1
    return [b for b in bonds if degree[b[0]] == 3 and degree[b[1]] == 3 and ring_count[b] == 1]


def is_bipartite(n, bonds):
    colour = -np.ones(n, int)
    colour[0] = 0
    stack = [0]
    adj = [[] for _ in range(n)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if colour[j] < 0:
                colour[j] = 1 - colour[i]
                stack.append(j)
            elif colour[j] == colour[i]:
                return False
    return True


def yml_text(name, args):
    pos, bonds, rings = build(name)
    n = len(pos)
    bridges = set(bridge_bonds(bonds, rings, n))
    hopping = [args.bridge_hopping if b in bridges else args.hopping for b in bonds]
    U = args.U_eV / args.t_eV
    h5 = args.h5file or f"{name}_b{args.beta:g}_Nt{args.Nt}.h5"

    lines = [
        f"# {name}: {n} sites, {len(bonds)} bonds, {len(rings)} rings, bipartite={is_bipartite(n, bonds)}",
        f"# ideal honeycomb geometry, bond length = 1 length unit ({args.length_unit} Angstrom)",
        f"# energies in units of t = {args.t_eV} eV: U = {args.U_eV} eV / t = {U:.12g}",
        "",
        "system:",
        f"  name: {name}",
        f"  nions: {n}",
        f"  Nt: {args.Nt}",
        f"  beta: {args.beta:g}",
        f"  U: {U:.17g}",
        "  mu: 0",
        "  offset: 0",
        "  positions:",
    ]
    lines += [f"  - [{x:.10f}, {y:.10f}, 0.0]" for x, y in pos]
    lines += ["  adjacency:"]
    lines += [f"  - [{i}, {j}]" for i, j in bonds]
    lines += ["  hopping:"]
    for b, h in zip(bonds, hopping):
        note = "   # inter-naphthalene bridge bond" if b in bridges else ""
        lines.append(f"  - {h:.12g}{note}")
    lines += [
        "",
        "PPP:",
        f"  form: {args.form}            # ohno | mataga-nishimoto | onsite | explicit",
        f"  hopping energy: {args.t_eV}   # eV",
        f"  length unit: {args.length_unit}      # Angstrom",
        f"  dielectric: {args.dielectric}",
        "",
        "HMC:",
        f"  autotune: {'true' if args.autotune else 'false'}   # true: run via autotuner8/autotuner.py",
        (f"  # seed: {args.seed}     # only used with autotune: false" if args.autotune
         else f"  seed: {args.seed}"),
        f"  Ntherm: {args.Ntherm}",
        f"  Nconf: {args.Nconf}",
        "  save frequency: 1",
        f"  Nradial: {args.Nradial}",
        f"  radial scale: {args.radial_scale:g}",
        "  Nhmc: 1",
        "",
        "Leapfrog:",
        f"  trajectory length: {args.trajectory_length!r}",
        f"  Fourier acceleration: {'true' if args.fourier_acceleration else 'false'}"
        + "   # exact Fourier acceleration (see README); false: unit-mass leapfrog",
        f"  Nmd: {args.Nmd}" + ("            # ignored with autotune: true (tuned, starting at 200)" if args.autotune else ""),
        "",
        "stability: QR",
        "",
        "measurements:",
        f"  Number Time Sources: {args.time_sources}",
        "",
        "fileIO:",
        f"  h5file: {h5}",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("molecule", choices=sorted(MOLECULES))
    p.add_argument("-o", "--output", help="yml file to write (default: stdout)")
    p.add_argument("--beta", type=float, default=4.0, help="inverse temperature in units of 1/t")
    p.add_argument("--Nt", type=int, default=16)
    p.add_argument("--t-eV", type=float, default=2.4, help="hopping energy t in eV")
    p.add_argument("--U-eV", type=float, default=8.0, help="on-site interaction in eV")
    p.add_argument("--form", default="ohno")
    p.add_argument("--dielectric", type=float, default=2.0)
    p.add_argument("--length-unit", type=float, default=1.4, help="bond length in Angstrom")
    p.add_argument("--hopping", type=float, default=1.0, help="hopping of every bond (units of t)")
    p.add_argument("--bridge-hopping", type=float, default=None,
                   help="hopping of bridge bonds (default: same as --hopping)")
    p.add_argument("--autotune", action="store_true", help="write `HMC: autotune: true`")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--Ntherm", type=int, default=200)
    p.add_argument("--Nconf", type=int, default=2000)
    p.add_argument("--Nmd", type=int, default=10)
    p.add_argument("--fourier-acceleration", action=argparse.BooleanOptionalAction, default=True,
                   help="exact Fourier acceleration (default); --no-fourier-acceleration: unit-mass leapfrog")
    p.add_argument("--trajectory-length", type=float, default=None,
                   help="MD trajectory length (default: pi/2 with Fourier acceleration, else 1)")
    p.add_argument("--Nradial", type=int, default=1)
    p.add_argument("--radial-scale", type=float, default=0.1)
    p.add_argument("--time-sources", type=int, default=4)
    p.add_argument("--h5file", default=None)
    args = p.parse_args(argv)
    if args.bridge_hopping is None:
        args.bridge_hopping = args.hopping
    if args.trajectory_length is None:
        args.trajectory_length = math.pi/2 if args.fourier_acceleration else 1.0

    text = yml_text(args.molecule, args)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
