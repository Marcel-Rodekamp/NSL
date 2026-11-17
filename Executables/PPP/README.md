# PPP simulations with NSL

## Model and conventions

```text
H = - Σ_{xy,σ} κ_xy c†_xσ c_yσ  +  ½ Σ_xy V_xy q_x q_y ,      q_x = n_x↑ + n_x↓ − 1 ,   V_xx = U
```

Energies are in units of the hopping energy `t` (`PPP: hopping energy`, in eV), so
`system: U`, `system: beta` (in 1/t) and the hopping list are dimensionless as for the
Hubbard runs. Site positions are in units of `PPP: length unit` (Å).

The interaction is decoupled in the **charge** channel on every time slice,
`S[φ] = ½ Σ_t φ_tᵀ (δV)⁻¹ φ_t` with `δ = β/Nt`
(`NSL::Action::PPPGaugeAction`), and the fermions use the charge-basis matrix
`NSL::FermionMatrix::HubbardExp` (field enters as `exp(±iφ)`). The spin-basis matrix
(`exp(±φ)`) cannot be used: with this gauge action it would simulate
`−½ Σ V_xy s_x s_y`, which agrees with PPP only for the on-site term
(details in [spin_basis_note.md](spin_basis_note.md)).
`V` must be positive definite; the action checks this.

## Interaction (`PPP` block of the yml file)

```yaml
PPP:
  form: ohno            # ohno | mataga-nishimoto | onsite | explicit
  hopping energy: 2.4   # t in eV                       (default 2.7)
  length unit: 1.4      # Angstrom per position unit    (default 1.4)
  dielectric: 2.0       # ε, applied to x ≠ y           (default 1.0)
  # e2: 14.399645       # e²/4πε0 in eV Å
  # V: [[...], ...]     # form: explicit only — nions × nions, units of t
```

With `r` the distance in Å and `U` in eV (all converted back to units of t):

| form               | `V_xy`, x ≠ y              |
|--------------------|----------------------------|
| `ohno`             | `U / sqrt(1 + (U ε r/e²)²)`|
| `mataga-nishimoto` | `U / (1 + U ε r/e²)`       |
| `onsite`           | `0` (Hubbard model)        |
| `explicit`         | given matrix               |

Both distance-dependent forms approach `e²/(ε r)` at large `r`. The code lives in
`src/NSL/Action/Implementations/PPP_Interaction.tpp`.

## Running

Inputs are in `yml/`; new molecules (ideal benzenoid geometries, per-bond hopping
lists) can be generated with `python/ppp_geometry.py`, e.g.

```sh
python python/ppp_geometry.py perylene --beta 40 --Nt 160 --autotune -o perylene_b40.yml
python python/ppp_geometry.py perylene --bridge-hopping 0.9 ...   # rescale the 2 inter-naphthalene bonds
```

`<build>` below is the build directory, `build/` if you follow *Build on macOS with Homebrew
(tested)* in the top-level [README](../../README.md).

* **Autotuned production run** (`HMC: autotune: true`, same protocol as
  `autotune_RadialMCMC`):
  `python autotuner8/autotuner.py yml/perylene.yml <build>/Executables/PPP/autotune_RadialMCMC_PPP`
* **Plain run** (`HMC: autotune: false`): uses `Leapfrog: Nmd`, `HMC: Ntherm`, `HMC: Nconf`
  and the optional `HMC: seed`; rerunning on an existing h5 file extends the chain.
  `<build>/Executables/PPP/autotune_RadialMCMC_PPP --file yml/naphthalene_benchmark.yml`
* **Measurements**: `<build>/Executables/Hubbard/propagator --file <yml> [--config N]`, with
  `measurements: basis: charge` (required; PPP ensembles exist only in the charge basis) and
  `type: spatial | momentum | both` (default: momentum if `wallSources` are given, otherwise
  spatial). `--config N` starts at configuration N; `--overwrite` recomputes existing results
  (the ensemble itself is never modified). Position-space propagators are stored as

  ```text
  <name>/markovChain/<n>/propagator/spatial/{particle,hole}/invM[t,<s>]   (Nt, Nx, Nx)
  <name>/markovChain/<n>/propagator/spatial/{particle,hole}/invM[<s>,t]
  <name>/markovChain/<n>/propagator/spatial/{particle,hole}/invM[t,t]
  ```

  for `Number Time Sources` source times s, and the free theory (phi = 0, s = 0) in
  `<name>/NonInteracting/propagator/spatial/...`. Momentum-projected propagators go to
  `.../propagator/momentum/<species>/invM[...]/<kSink>-<kSrc>` (all pairs of the wall sources,
  each followed by its complex conjugate) and `.../propagator/momentum/<species>/Q`
  (1/Nt sum_t Tr invM[t,t]), with the sources used in `<name>/Meta/propagator/momentum/wallSources`.
  Charge and spin basis measure the same quantities; see `Executables/Hubbard/propagator.cpp`.
  `python/ppp_compare.py` and `python/ppp_autocorr.py` read this layout and the older one
  (`propagator/{particle,hole}/...`, from the former `propagatorChargeBasis`).

The meta data of an ensemble contains the interaction (`<name>/Meta/PPP/{form,V,...}`).

Slides explaining these changes, the spin-basis restriction, the ED benchmarks, exact Fourier
acceleration and the bugs fixed along the way are in [notes/](notes/index.qmd)
(Quarto revealjs; `quarto render` in that folder writes `notes/_slides/`).

## Exact Fourier acceleration

```yaml
Leapfrog:
  trajectory length: 1.5707963267948966   # pi/2
  Fourier acceleration: true   # default false: unit-mass leapfrog
  Nmd: 10
```

switches the molecular dynamics from the unit-mass leapfrog (`LeapfrogRealForce`) to
`NSL::Integrator::ExactFourierAcceleration`
(`src/NSL/Integrator/Impl/exactFourierAcceleration.tpp`, after J. Ostmeyer and
P. Buividovich, [arXiv:2404.09723](https://arxiv.org/abs/2404.09723)):

* the mass matrix is the gauge action's quadratic form, so the momenta are drawn from
  `p_t ~ N(0, (δV)⁻¹)` and `K = ½ Σ_t p_tᵀ (δV) p_t`;
* the gauge action is integrated exactly. Every mode of `φ` rotates with unit frequency,
  `φ → φ cos τ + δV p sin τ`, and only the fermion force is integrated numerically
  (`R(ε/2) K(ε) R(ε/2)`, i.e. `Nmd` fermion forces per trajectory instead of `Nmd+1`);
* without it the gauge modes oscillate with `ω_k = (δλ_k)^{-1/2}`, `λ_k` the eigenvalues
  of `V`. For perylene at `δ = 1/4` these range from 0.44 (uniform charge mode,
  `λ = 20.7`) to 2.15 (`λ = 0.87`): a trajectory of length 1 rotates the uniform mode by
  only 0.44 rad, while the step size is limited by the stiff staggered modes.
  "Uniform charge mode" (my label) is the field along the eigenvector `v_0` of `V` with the
  largest eigenvalue. `v_0` is 98.7 % the uniform vector, so this mode couples to the total
  charge; "staggered" modes alternate in sign between the sublattices. `a_k`, the
  observables in `python/ppp_autocorr.py`, are the time-summed field along `v_k`, normalized
  to unit variance for the gauge action alone (notes deck 4).

`Nmd` counts steps as before and is tuned by `autotuner.py` in the same way (tested on
benzene); at equal acceptance it is about half the leapfrog value. The default leapfrog
path is unchanged (same random numbers, bit-identical chains). The integrator only evolves
the real part of `φ` and keeps a constant imaginary offset, like `LeapfrogRealForce`. It
also works for the Hubbard gauge action, with `A = 1/(δU)·1` passed as the mass matrix.

**Recommended: `Fourier acceleration: true` with `trajectory length: 1.5707963` (π/2)**, the
optimum for the Gaussian part (every gauge mode is rotated by exactly π/2). `yml/perylene.yml`
and `python/ppp_geometry.py` use this by default.

Benchmarks (2026-11-16, β = 20, Nt = 80, screened Ohno, Nradial = 1, Release build, 3–4
chains running in parallel on an Apple M-series laptop). Cost per independent sample is
2τ·(seconds per trajectory), where τ is the larger of the Γ-method and binned estimates of
the integrated autocorrelation time:

| molecule    | integrator, T, Nmd      | acceptance | s/traj | τ G(1–4) | cost G   | slowest observable, τ | cost slowest |
|-------------|-------------------------|------------|--------|----------|----------|-----------------------|--------------|
| naphthalene | leapfrog, 1, 7          | 0.78       | 0.048  | 4.8      | 0.45 s   | S_fermion, 13         | 1.27 s       |
| naphthalene | leapfrog, π/2, 10       | 0.74       | 0.063  | 3.0      | 0.38 s   | S_fermion, 18         | 2.3 s        |
| naphthalene | EFA, 1, 3               | 0.80       | 0.023  | 2.1      | 0.10 s   | a_6², 2.3             | 0.11 s       |
| naphthalene | **EFA, π/2, 5**         | 0.81       | 0.032  | 1.1      | 0.073 s  | a_9², 3.1             | 0.20 s       |
| perylene    | leapfrog, 1, 12         | 0.90       | 0.284  | 5.1      | 2.9 s    | a_0, 19               | 11.0 s       |
| perylene    | EFA, 1, 6               | 0.87       | 0.143  | 1.9      | 0.55 s   | S_fermion, 16         | 4.7 s        |
| perylene    | **EFA, π/2, 10**        | 0.92       | 0.226  | 0.74     | 0.33 s   | a_18², 2.4            | 1.07 s       |

With T = π/2, Fourier acceleration is 6–9 times cheaper per independent propagator sample
and 6–10 times cheaper for the slowest observable. With the leapfrog the uniform charge mode
`a_0` (largest eigenvalue of V) is slowest, as expected from `ω_0 = 0.44`. Second perylene
chains (different seed) gave τ G(1–4) = 0.9–1.0 for EFA T = 1 and 0.6–1.1 for EFA T = π/2,
so the two EFA trajectory lengths perform similarly per CPU second; π/2 decorrelates the
propagator fastest per trajectory. Plots: `figures/efa_autocorrelation_*.pdf`.

**Rare excursions, all integrators.** Occasionally a chain enters a region with a larger
fermion action (smaller |det M|) and stays there for a few hundred to ~2000 trajectories:
seen in both naphthalene leapfrog runs, in one of the two perylene EFA T = 1 chains (right
after thermalization, `S_fermion` ≈ 3–4 above its mean for ~2000 trajectories) and, weaker,
in one perylene EFA π/2 chain. These episodes give `S_fermion` a slow tail in its
autocorrelation that the automatic Γ window misses. For production: thermalize generously
(a few thousand trajectories are cheap with EFA), watch `S_fermion`, run several
independent chains and estimate errors with bins of ≥ 500 trajectories or from the spread
between chains.

Correctness: the naphthalene ensembles of all four integrators agree with exact
diagonalization (correlated χ² p = 0.19–0.50). For perylene (no exact result) the five
chains (leapfrog, 2× EFA T = 1, 2× EFA π/2, 5000 trajectories each) agree on the action
except the EFA T = 1 chain with the long early excursion; the propagators G(1–20) agree
with p = 0.02–0.35 using errors from 10 bins of 500. That is a somewhat larger spread than
those errors imply, consistent with the rare excursions above (and with 10-bin errors
being uncertain by ~25%). Unit tests: `Tests/Integrator/test_exactFourierAcceleration.cpp`
(exact free flow, reversibility, offset kept fixed, second order energy violation,
phase-space volume, momentum distribution).

**Disk space:** the spatial propagator measurement stores `invM[t,s]`, `invM[s,t]` and `invM[t,t]` for
every time source and species, about 8.6 MB per perylene configuration at Nt = 80 with 4
time sources (43 GB for 5000 configurations; naphthalene 2.1 MB). Plan storage accordingly,
or reduce `Number Time Sources`.

`python/ppp_autocorr.py run.yml [run.h5] --seconds-per-trajectory s` reports integrated
autocorrelation times of the action, of the field amplitudes `a_k` along the eigenvectors
of `V` and, if measured, of the propagator, from the Γ method (automatic windowing) and
from bin averages. A `!` marks observables whose binned estimate is much larger: the
automatic window misses rare long excursions, so use the larger value for error bars.

## Benchmarks against exact diagonalization

Integrating out φ slice by slice is exact, so the ensemble averages of the measured
`M⁻¹` equal the propagators of the transfer matrix `W = exp(δK̂) exp(−δH_V)` at the *same*
`Nt`; `python/ppp_ed.py` computes these exactly (no Trotter extrapolation needed).

```sh
python python/ppp_ed.py --self-test                 # checks the operator mapping against NSL's fermion matrix
python python/ppp_ed.py run.yml -o run_ed.h5        # dense (≤ 8 sites) or low-lying (larger) backend
python python/ppp_compare.py run.yml run.h5 run_ed.h5 --plot run.pdf
```

`ppp_compare.py` first checks NSL's measured `invM` against a dense inverse of the fermion
matrix for a few stored configurations, then compares the time-source averaged ensemble
propagators (binned jackknife errors) with the exact ones.

The benchmark inputs `yml/benzene_benchmark.yml` (β = 12) and
`yml/naphthalene_benchmark.yml` (β = 20, βΔ ≈ 90 with Δ the bandwidth) use a screened Ohno
interaction (t = 2.4 eV, U = 8 eV, ε = 2, bond length 1.4 Å): a standard parameter set for
π-conjugated hydrocarbons with sizeable long-range interactions (V_nn ≈ 0.54 U) and a
well-conditioned V (condition number ≈ 10–24 for these molecules), which keeps the HMC
efficient. Unit tests: `Tests/Action/test_PPPGaugeAction.cpp`.

Validation (2026-11-16, 10–20k configurations each): benzene β = 4 (Ohno and
Mataga–Nishimoto, two seeds each), benzene β = 12 and naphthalene β = 20 all agree with the
exact lattice propagators (correlated χ² p-values 0.27–0.73). The same data are
incompatible with the on-site-only (Hubbard) interaction (⟨z²⟩ ≈ 400–650), so the test
resolves the long-range part of V. The comparison plots and run settings are in
[figures/](figures/README.md).
