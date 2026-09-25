# Dataset-only workflow

This page walks through the "no callable model, just a fixed dataset" case: you have a
table of observed `(X, Y)` pairs — a factor sample and an output that came with it — and
want each factor's Shapley effect, without being able to run a model at new points.

## Why this is different from the model-driven workflow

[Getting started](@ref) and [How it works](@ref) cover the case where you have a callable
model and can evaluate it at any input you construct. Here, `Y` is fixed data, not something
you can compute on demand — so the estimator can only reuse output values that are already
present in the dataset. It does this via a nearest-neighbour lookup instead of an exact
function call: to estimate the effect of holding a subset of factors fixed, it finds the
dataset row whose values on those factors are closest to a reference point, and uses that
row's already-known `Y`.

This costs **zero new evaluations** — appropriate when there is no model to call, or when
calling one is prohibitively expensive — at the cost of some estimator variance from the
nearest-neighbour approximation itself, on top of the usual Monte Carlo sampling variance.

## 1. Build (or load) the dataset

Any table of `(X, Y)` observations works. Here, one is generated from the
[Ishigami function](https://en.wikipedia.org/wiki/Ishigami_function) for illustration —
in practice `X` and `Y` would usually already exist as recorded data.

```@example dataset_only
using SAShE, DataFrames, Distributions, Random
Random.seed!(2468)

function ishigami(x)
    a, b = 7.0, 0.1
    return (1 + b * x[3]^4) * sin(x[1]) + a * sin(x[2])^2
end

d = Uniform(-π, π)
n_samples, n_factors = 5000, 3
factor_names = [:x1, :x2, :x3]

X = DataFrame(rand(d, n_samples, n_factors), factor_names)
Y = map(row -> ishigami(collect(row)), eachrow(X))
nothing # hide
```

## 2. Wrap it in a `DataModel` and analyze

```@example dataset_only
model = DataModel(X, Y)
Φₙ, Φ²ₙ = analyze(model, 5000, PickAndFreeze())
nothing # hide
```

The second argument to `analyze` is the number of random permutations to average over — an
accuracy/budget knob, distinct from `n_samples` above (the size of the dataset itself). The
third, `PickAndFreeze()` — see [How it works](@ref) — must be given explicitly; there is no
default, and for now it's the only estimator implemented for `DataModel`. `Φₙ` and `Φ²ₙ` are
increments in the same shape `CallableModel`'s workflow returns — every downstream helper
works unchanged.

## 3. Turn the increments into estimates

```@example dataset_only
Φ, Φlb, Φub = SAShE.shapley_effects(Φₙ, Φ²ₙ)
[Φlb Φ Φub]
```

## 4. `sum(Φ)` vs. the variance

```@example dataset_only
sum(Φ), var(Y)
```

Unlike the equivalent step for `CallableModel` (see [Getting started](@ref)), this is **not**
a sanity check you can act on: `analyze`'s random-permutation walk ends every permutation at
`var(Y)` directly, computed from the dataset rather than from the nearest-neighbour
estimator being tested, so `sum(Φ)` equals `var(Y)` exactly (up to floating point) no matter
how many permutations you use or how good the nearest-neighbour lookups are — see
[Deliberate deviations](@ref). A mismatch here points to a bug in this package, not in your
data or parameters.

## Caveats

- **The neighbour count is fixed** — one neighbour besides the query point itself, which is
  [1]'s `N_I = 2` (§6.1.2) — not a tunable accuracy knob. `V_u = Var(E(Y|X_u))`
  is recovered via `E(E(Y|X_u)²) = E(f(X)f(X^u))` ([1]'s Proposition 2), which holds only
  because it multiplies exactly *two* conditionally-independent draws sharing the same
  conditional mean `Z = E(Y|X_u)` — the standard `E[Z₁Z₂] = Z²` trick for an unbiased
  estimator of a squared mean (`E[Z₁²]` alone would be biased upward by `Var(Z₁)`). One of
  the two draws is the query point's own, exact output `Y[s]`; the other is its nearest
  neighbour's. Multiplying three or more draws together would estimate a different, wrong
  quantity, not a more accurate `V_u`. Increasing dataset size or permutation count are the
  available ways to reduce estimator variance instead.
- **Distance is standardized Euclidean** by default (each coordinate divided by its own
  standard deviation before comparing — divided, not centred, since Euclidean distance is
  translation-invariant, so the coordinates are not z-scores) — unweighted Euclidean distance
  would otherwise let whichever coordinate has the largest variance dominate the search. This
  accounts for scale but not correlation between coordinates (a Mahalanobis-distance option,
  accounting for both, is planned).
- **The confidence intervals are anti-conservative here.** `margin_of_error` assumes each
  permutation is an independent draw, which [1] §6.2 explicitly notes is false for this
  estimator — every permutation reuses the same dataset. Adding permutations shrinks the
  reported interval but not the nearest-neighbour error floor it cannot see, so treat the
  interval as a lower bound on uncertainty. See [Deliberate deviations](@ref).
- Only continuous/discrete numerical inputs are supported; categorical inputs are not yet.
- This estimator's convergence and the assumptions behind it are set out in [1] §6 — in
  particular Theorem 4 for its rate, and §6.2 for how the conditional elements are aggregated.
  See the [References](@ref) page.

See the [References](@ref) page for the estimator's full citation.
