# Particle/hole basis and the sublattice sign

NSL's charge basis (`HubbardExp`) uses particles and holes *without* a sublattice sign. This page states the
convention, shows which symmetry generator carries the sublattice sign as a result, and gives the multi-fermion
operators of definite isospin and spin that follow from it. Read it before building interpolating operators with
isospin or spin labels.

## The basis

```text
p_x = c_{x↑}        h_x = c†_{x↓}        (no sublattice sign)

Q_x = p†_x p_x − h†_x h_x = n_{x↑} + n_{x↓} − 1        (charge)
```

In `NSL::FermionMatrix::HubbardExp`, `Species::Hole` sets `sgn = −1`, which multiplies the field, the hopping and the
chemical potential: holes see `exp(−iφ)`, `exp(−δK)` and `exp(−δμ)`. The hopping changes sign because, for down
electrons,

```text
−κ c†_{x↓} c_{y↓} = +κ h†_y h_x .
```

The stored hole propagator is `⟨h_x h†_y⟩ = ⟨c†_{x↓} c_{y↓}⟩`, with the same time ordering as for particles.

`HubbardExpSpinBasis` works with spin-up and spin-down electrons. The electron-basis forms below apply to it.

## Spin and isospin

On a bipartite lattice, let `(−1)^x` be +1 on one sublattice and −1 on the other.

```text
spin      S⁺  = Σ_x p†_x h†_x                       = Σ_x c†_{x↑} c_{x↓}
          S_z = ½ Σ_x (p†_x p_x + h†_x h_x − 1)       = ½ Σ_x (n_{x↑} − n_{x↓})

isospin   I⁺  = Σ_x (−1)^x p†_x h_x                  = Σ_x (−1)^x c†_{x↑} c†_{x↓}
          I_z = ½ Σ_x (p†_x p_x − h†_x h_x)           = Q/2
```

`I⁺` is Yang's η-pairing operator. The lowering operators are the adjoints, `S⁻ = (S⁺)†` and `I⁻ = (I⁺)†`.

**The sublattice sign sits in the isospin generators.** In any basis where p and h are related site by site, exactly
one of the two SU(2)s carries `(−1)^x`. Defining holes as `h_x = (−1)^x c†_{x↓}` would move the sign into `S^±`; it
cannot remove it from both. NSL keeps spin uniform, as in the electron basis, for two reasons:

| symmetry      | holds for                                                              |
|---------------|------------------------------------------------------------------------|
| spin SU(2)    | every lattice, any μ, Hubbard and PPP                                  |
| isospin SU(2) | bipartite lattices at μ = 0 with an on-site interaction only (Hubbard) |

- μ ≠ 0 leaves only `I_z`, i.e. the charge.
- The PPP interaction `½ Σ_xy V_xy Q_x Q_y` with `x ≠ y` breaks isospin SU(2). On a bipartite lattice at μ = 0 it
  keeps `I_z` and the π rotation `exp(iπ I_y)`, which maps `I_z → −I_z`. This π rotation is the particle-hole
  transformation `c_{xσ} → (−1)^x c†_{xσ}` combined with a π spin rotation. In quantum chemistry it is the alternancy
  symmetry of alternant hydrocarbons, which gives the ± labels of their states.

So the sign sits on the symmetry that needs a bipartite lattice anyway, and spin, which holds everywhere, stays
uniform.

**Converting.** The other convention changes the sign of hole propagators between opposite sublattices:
`G^h_{xy} → (−1)^{x+y} G^h_{xy}`. Diagonal and same-sublattice elements do not change. Use this factor when comparing
with code that defines `h_x = (−1)^x c†_{x↓}`.

`NSL::Lattice::bipartite()` tells whether the sign exists. The lattice classes do not expose the sign itself yet.

## The single-fermion multiplet

At each site the four fermion operators form one multiplet with `(I, S) = (½, ½)`. With Condon–Shortley phases (the
components follow from `p†_x` by applying `I⁻` and `S⁻`):

```text
(I_z, S_z)     component
(+½, +½)       p†_x
(+½, −½)       h_x
(−½, +½)       (−1)^x h†_x
(−½, −½)       −(−1)^x p_x
```

The isospin partner of `p†_x` carries the sublattice sign. A table that lists `p†, h, h†, p` as the four components,
without the sign, gives the right spin and the wrong isospin.

## Orbital labels and partners

An operator with label k is built with the same amplitudes for all four operator types:

```text
X_k = Σ_x u_k(x) X_x ,        X = p, p†, h, h† .
```

The **partner** k̄ of k is the label with amplitudes `(−1)^x u_k(x)`:

- for a site, `X_{x̄} = (−1)^x X_x`;
- for an eigenorbital of a bipartite hopping matrix with energy `ε_k`, k̄ is its chiral partner, with energy `−ε_k`.
  On the honeycomb it is the other band at the same momentum. With the eigenvectors sorted by energy and their
  phases chosen so that `u_{N−1−k}(x) = (−1)^x u_k(x)`, the partner of k is `N−1−k`.

The single-fermion multiplet of orbital k is `(p†_k, h_k, h†_k̄, −p_k̄)`.

## Two-fermion operators

Coupling two multiplets with Clebsch–Gordan coefficients gives the operators below, up to an overall sign per entry.
**In every term, each charge-lowering factor (p or h†) carries the partner label.** The `I_z = +1` entries have no
such factor; in the `I_z = −1` entries the bars only relabel `(k, l) → (k̄, l̄)`. With the bars, each column is one
multiplet, so the ladder operators map an entry to its neighbour.

```text
I = 1, S = 1
I_z = +1     S_z = +1     p†_k p†_l
             S_z =  0     (p†_k h_l + h_k p†_l)/√2
             S_z = −1     h_k h_l
I_z =  0     S_z = +1     (p†_k h†_l̄ + h†_k̄ p†_l)/√2
             S_z =  0     ½(p_k̄ p†_l + p†_k p_l̄ − h_k h†_l̄ − h†_k̄ h_l)
             S_z = −1     (p_k̄ h_l + h_k p_l̄)/√2
I_z = −1     S_z = +1     h†_k̄ h†_l̄
             S_z =  0     (p_k̄ h†_l̄ + h†_k̄ p_l̄)/√2
             S_z = −1     p_k̄ p_l̄

I = 1, S = 0
I_z = +1     (p†_k h_l − h_k p†_l)/√2
I_z =  0     ½(p†_k p_l̄ − p_k̄ p†_l + h_k h†_l̄ − h†_k̄ h_l)
I_z = −1     (p_k̄ h†_l̄ − h†_k̄ p_l̄)/√2

I = 0, S = 1 (I_z = 0)
S_z = +1     (p†_k h†_l̄ − h†_k̄ p†_l)/√2
S_z =  0     ½(p_k̄ p†_l − p†_k p_l̄ + h_k h†_l̄ − h†_k̄ h_l)
S_z = −1     (p_k̄ h_l − h_k p_l̄)/√2

I = 0, S = 0 (I_z = 0)
             ½(p†_k p_l̄ + p_k̄ p†_l + h_k h†_l̄ + h†_k̄ h_l)
```

Without the bars, the `I_z = 0` entries have the stated isospin only when k and l are on the same sublattice. For
opposite sublattices `I = 0` and `I = 1` are swapped, and for orbital labels the operators mix both. The table was
checked with the exact isospin and spin Casimirs, for random complex amplitudes.

## Vacuum overlap

A correlator `⟨O(τ) O†(0)⟩` approaches the constant `|⟨O⟩|²` at large τ only when `O` has the quantum numbers of the
ground state; such a channel needs the vacuum subtraction.

- Hubbard model on a bipartite lattice at half filling (ground state `I = S = 0`): only the `I = 0, S = 0` channel.
- PPP on a bipartite lattice at μ = 0: under the π rotation above, the `I_z = 0` operators with `I = 0` are even and
  those with `I = 1` are odd. Odd operators have no vacuum expectation value, so again only `I = 0, S = 0` needs the
  subtraction.
- With translations or point-group symmetries, only operators with zero total momentum and in the totally symmetric
  representation can overlap with the vacuum.
