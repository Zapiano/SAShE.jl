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
Φₙ, Φ²ₙ = analyze(model, 5000)
nothing # hide
```

The second argument to `analyze` is the number of random permutations to average over — an
accuracy/budget knob, distinct from `n_samples` above (the size of the dataset itself).
`Φₙ` and `Φ²ₙ` are increments in the same shape `CallableModel`'s workflow returns — every
downstream helper works unchanged.

`analyze` also takes an `estimator=` keyword — see [How it works](@ref) — currently only
`PickAndFreeze()` (the default, so nothing above needs to change).

## 3. Turn the increments into estimates

```@example dataset_only
Φ, Φlb, Φub = SAShE.shapley_effects(Φₙ, Φ²ₙ)
[Φlb Φ Φub]
```

## 4. Sanity check: do they sum to the variance?

```@example dataset_only
sum(Φ), var(Y)
```

If these disagree by more than a few percent, increase the number of permutations (or, if
the dataset itself is small relative to `n_factors`, the nearest-neighbour lookups may be
too coarse — a larger dataset helps here too).

## Caveats

- **The neighbour count is fixed at 2**, not a tunable accuracy knob. `V_u = Var(E(Y|X_u))`
  is recovered via `E(E(Y|X_u)²) = E(f(X)f(X^u))` ([1]'s Proposition 2), which holds only
  because it multiplies exactly *two* conditionally-independent draws sharing the same
  conditional mean `Z = E(Y|X_u)` — the standard `E[Z₁Z₂] = Z²` trick for an unbiased
  estimator of a squared mean (`E[Z₁²]` alone would be biased upward by `Var(Z₁)`).
  Multiplying three or more draws together would estimate a different, wrong quantity, not
  a more accurate `V_u`. Increasing dataset size or permutation count are the available ways
  to reduce estimator variance instead.
- **Distance is standardized Euclidean** by default (each coordinate z-scored before
  comparing) — unweighted Euclidean distance would otherwise let whichever coordinate has
  the largest variance dominate the search. This accounts for scale but not correlation
  between coordinates (a Mahalanobis-distance option, accounting for both, is planned).
- Only continuous/discrete numerical inputs are supported; categorical inputs are not yet.

See the [References](@ref) page for the estimator's full citation.
