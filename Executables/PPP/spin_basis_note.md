# Why the PPP model cannot be simulated in the spin basis

**Short answer.** The spin basis rewrites the Hubbard interaction as a spin–spin
interaction. It does so with an identity that holds only *on a single site*. The
off-diagonal (x ≠ y) terms of the PPP interaction have no such identity. A spin-channel
auxiliary field combined with the PPP gauge action therefore simulates a different
Hamiltonian: a long-range *ferromagnetic Ising* model in place of the long-range *charge*
repulsion. Only the charge basis (`HubbardExp`, field enters as `exp(±iφ)`) reproduces PPP.

## Setup

```text
H_V = ½ Σ_xy V_xy q_x q_y ,   q_x = n_x↑ + n_x↓ − 1 ,   s_x = n_x↑ − n_x↓ ,   V_xx = U
```

## On-site term: both channels work

On a single site, `n ∈ {0,1}` implies `q_x² + s_x² = 1`, so

```text
½ U q_x²  =  ½ U  −  ½ U s_x²
```

and the Hubbard interaction can be decoupled in either channel (δ = β/Nt):

| basis  | NSL matrix            | identity                                              |
|--------|-----------------------|-------------------------------------------------------|
| charge | `HubbardExp`          | ∫dφ e^{−φ²/2δU + iφq} ∝ e^{−δU q²/2}                   |
| spin   | `HubbardExpSpinBasis` | ∫dφ e^{−φ²/2δU + φs} ∝ e^{+δU s²/2}                    |

The two right-hand sides differ only by a constant.

## Long-range term: only the charge channel works

With the PPP gauge action `S = ½ Σ_t φ_tᵀ (δV)⁻¹ φ_t`, on each time slice:

```text
charge:  ∫dφ e^{−½ φᵀ(δV)⁻¹φ + iφ·q}  ∝  e^{−½ δ qᵀVq}   →  H_V                       ✓
spin:    ∫dφ e^{−½ φᵀ(δV)⁻¹φ +  φ·s}  ∝  e^{+½ δ sᵀVs}   →  H_spin = −½ Σ_xy V_xy s_x s_y
```

The difference is

```text
H_spin − H_V = −½ Σ_xy V_xy (s_x s_y + q_x q_y)
```

* x = y: `s_x² + q_x² = 1`, a constant (−U/2 per site). This is why Hubbard works in both bases.
* x ≠ y: `s_x s_y + q_x q_y = 2(n_x↑ n_y↑ + n_x↓ n_y↓) − n_x − n_y + 1`, an operator.
  No constant shift or change of sign removes it.

Physically, PPP has charges on different sites interacting through `V_xy` regardless of spin.
The spin-channel model has *no* interaction between charges on different sites. Instead it has
a ferromagnetic coupling `−V_xy s_x s_y`.

## Two-site illustration

Two electrons on two sites, `V_12 = V`; interaction energies with the common constant removed:

| state      | q       | s       | PPP     | spin channel |
|------------|---------|---------|---------|--------------|
| ↑ , ↑      | (0, 0)  | (1, 1)  | 0       | −V           |
| ↑ , ↓      | (0, 0)  | (1, −1) | 0       | +V           |
| ↑↓ , 0     | (1, −1) | (0, 0)  | U − V   | U            |

In PPP the ionic state is lowered by V (electron–hole attraction, i.e. exciton binding), and
the spin orientation does not matter. The spin-channel model loses the exciton binding and
favours parallel spins by 2V.

## What this does to real molecules

Exact diagonalization at half filling with the benchmark parameters (screened Ohno,
t = 2.4 eV, U = 8 eV, ε = 2, bond length 1.4 Å). The table gives the lowest energy in each
S_z sector, relative to each model's ground state, in units of t:

| S_z | benzene PPP | benzene spin channel | naphthalene PPP | naphthalene spin channel |
|-----|-------------|----------------------|-----------------|--------------------------|
| 0   | 0.000       | 19.98                | 0.000           | 46.14                    |
| 1   | 1.458       | 18.30                | 1.030           | 44.01                    |
| 2   | 3.171       | 11.21                | 2.695           | 39.09                    |
| 3   | 5.853       | **0.000**            | 4.657           | 30.00                    |
| 4   |             |                      | 6.886           | 17.16                    |
| 5   |             |                      | 10.048          | **0.000**                |

PPP has a singlet ground state with a triplet gap of about 1 t. The spin-channel model
has a *fully polarized* ground state, with the singlet tens of t higher. Because V decays only
like 1/r, the ferromagnetic coupling `½ Σ_{x≠y} V_xy` summed over all pairs is large, so the
error grows with system size rather than being a small correction.

## Could the spin basis be repaired?

* **Imaginary coupling in the spin channel.** `e^{iφ·s}` gives `e^{−½ δ sᵀVs}`, an
  *antiferromagnetic* long-range Ising model. This is still not PPP.
* **Mixed basis (works, not implemented).** Split `V = (V − α𝟙) + α𝟙` with
  `0 ≤ α < λ_min(V)`. The first part is positive definite and goes into the charge channel.
  The second part is purely on-site and can use the spin channel through `q² = 1 − s²`. This
  needs two auxiliary fields; see e.g. Buividovich et al., PRB 98, 235129 (2018) for the
  extended Hubbard model. For perylene with the benchmark parameters,
  λ_min(V) = 0.87 t against U = 3.33 t, so at most ~26% of U could be moved to the spin channel.

## The charge basis costs nothing here

For a bipartite molecule such as perylene at μ = 0 with real φ, `det M_h = (det M_p)*`, so the
fermion weight `|det M_p|² ≥ 0` has no sign problem. The known ergodicity issue of the charge
basis (zeros of det M, Wynen et al., PRB 100, 075141 (2019)) is what NSL's radial updates
address (Temmen et al., PRB 112, 045134 (2025)).

## Safeguards in NSL

* `PPPGaugeAction` documents the restriction, and the PPP driver (`autotune_RadialMCMC_PPP`) is
  hard-wired to `HubbardExp`.
* `HubbardFermionAction` selects the basis at compile time from the fermion-matrix type.
  `populate(phi, bool reshape)` accepts only a genuine `bool`. Passing a `Species` to a
  spin-basis matrix, or a `Spin` to a charge-basis matrix, is therefore a compile error and no
  longer converts silently.
