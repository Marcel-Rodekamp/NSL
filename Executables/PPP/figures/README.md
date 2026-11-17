# Figures

## NSL vs exact diagonalization

Each figure shows the time-source averaged propagators `G_xy(t)` measured by NSL (points, binned
jackknife errors) against the exact lattice propagators at the same `Nt` from `python/ppp_ed.py`
(line), for particles and holes. Made with
`python/ppp_compare.py <yml> <run.h5> <run_ed.h5> --plot <figure.pdf>` on 2026-11-16.

All runs use screened Ohno or Mataga–Nishimoto interaction with t = 2.4 eV, U = 8 eV, ε = 2,
bond length 1.4 Å, plain HMC mode (`HMC: autotune: false`), Ntherm = 500.

| figure                  | molecule    | form             | β  | Nt | Nmd | Nconf | seed | χ²/dof | p    |
|-------------------------|-------------|------------------|----|----|-----|-------|------|--------|------|
| `benzene_b4.pdf`        | benzene     | ohno             | 4  | 16 | 4   | 20000 | 2026 | 1.05   | 0.63 |
| `benzene_b4_rep.pdf`    | benzene     | ohno             | 4  | 16 | 4   | 20000 | 777  | 0.98   | 0.73 |
| `benzene_b4_MN.pdf`     | benzene     | mataga-nishimoto | 4  | 16 | 4   | 20000 | 2026 | 1.66   | 0.02 |
| `benzene_b4_MN_rep.pdf` | benzene     | mataga-nishimoto | 4  | 16 | 4   | 20000 | 777  | 0.99   | 0.72 |
| `benzene_b12.pdf`       | benzene     | ohno             | 12 | 48 | 6   | 20000 | 2026 | 1.27   | 0.27 |
| `naphthalene_b20.pdf`   | naphthalene | ohno             | 20 | 80 | 12  | 10000 | 2026 | 1.20   | 0.37 |

χ²/dof and p are from the correlated χ² over distance classes × a subset of time slices that
`ppp_compare.py` reports. The p = 0.02 run is a statistical fluctuation: the same setup with
another seed gives p = 0.72.

Inputs: `benzene_b12` and `naphthalene_b20` are `yml/benzene_benchmark.yml` and
`yml/naphthalene_benchmark.yml`. The β = 4 runs are `yml/benzene_benchmark.yml` with
`beta: 4`, `Nt: 16`, `Nmd: 4`, and, where listed, `PPP: form: mataga-nishimoto` and/or
`HMC: seed: 777`.

The log-scale panels for `G_00` show the large relative errors near t ≈ Nt/2. This is the
known signal-to-noise limit of charge-basis propagators, not a disagreement with ED.

## Exact Fourier acceleration vs leapfrog

`efa_autocorrelation_naphthalene.pdf` and `efa_autocorrelation_perylene.pdf` show the
normalized autocorrelation functions of the site-averaged propagator `G(τ = δ)` and of the
uniform charge mode `a_0` (time-summed field along the eigenvector of V with the largest
eigenvalue). The horizontal axis is CPU time (Markov time × seconds per trajectory), so
the curves compare the cost of decorrelating. Dashed: unit-mass leapfrog, solid: exact
Fourier acceleration (EFA). Each setting keeps its color in both figures.

Runs: β = 20, Nt = 80, screened Ohno (t = 2.4 eV, U = 8 eV, ε = 2), Nradial = 1, radial
scale 0.1, Ntherm = 500; naphthalene 8000 and perylene 5000 trajectories per setting, 4 time
sources for the propagator. Integrated autocorrelation times and costs are in the table in
`../README.md` (section *Exact Fourier acceleration*), computed with `python/ppp_autocorr.py`.
