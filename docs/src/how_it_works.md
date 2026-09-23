# How it works

You can use SAShE without reading this page. It is here for when you want to know what
`analyze` is doing, or when you need finer control over the sampling — running the model on
a cluster, checkpointing, or handling dependent factors.

## The idea: pick-freeze

A Shapley effect asks: *as we reveal the value of factor `j` on top of an already-known
subset of the other factors, how much does that shrink the remaining uncertainty in the
output — averaged over every possible "already-known" subset?*

Estimating this by Monte Carlo needs, for many random subsets `u` of the factors:

- `f(x)` — every factor at its reference value (a row of `X1`)
- `f(x_u, y₋ᵤ)` — factors in `u` held at reference, the rest resampled (from `X2`)
- `f(x_{u+j}, y₋₍ᵤ₊ⱼ₎)` — the same, but with `j` now also held

The change the extra held factor makes, suitably weighted and averaged, is the estimator.
Goda's contribution ([4], see [References](@ref)) is to walk a **random permutation** of
the factors and move them one at a time from the resampled set to the held set: a single
pass yields the term for every factor at once, at `d + 1` model evaluations per sample
instead of `3d`.

## The `estimator=` keyword

`analyze` takes an `estimator=` keyword selecting the estimation method — currently only
`PickAndFreeze()` (the algorithm this page describes), which is also the default, so
existing calls don't need to change. It exists as an extension point: other estimation
methods are expected to land behind the same keyword later, without changing how you build
or call `CallableModel`, `PickAndFreezeSample`, or `DataModel`.

```julia
Φₙ, Φ²ₙ, Yₙ = analyze(model; estimator=PickAndFreeze())   # equivalent to analyze(model)
```

## Permutations

For `N` base samples SAShE draws an `N × d` matrix `π`, one random factor ordering per row.
`generate_permutations(N, d)` produces it; `CallableModel` and `PickAndFreezeSample` do this for you
and store it, so the sampling and the analysis always use the same orderings.

## What is `Z`?

`Z` is the matrix of factor values the model is actually evaluated at. For `N` base samples
and `d` factors it has **`N · (d + 1)` rows**. Each base sample contributes:

- one row that is the untouched `X1` sample, then
- `d` rows in which factors are progressively taken from `X2` instead of `X1`, following
  that sample's permutation.

For `N = 2`, `d = 3`, with permutations `[3 1 2]` and `[2 1 3]` (writing **A** for a value
taken from the `X1` row and **B** for one taken from the `X2` row):

| block | row | x1 | x2 | x3 | taken from `X2` |
| :---: | :-: | :-: | :-: | :-: | :-------------- |
| 1 | 1 | A | A | A | — (base sample) |
| 1 | 2 | A | A | B | `x3` |
| 1 | 3 | B | A | B | `x1, x3` |
| 1 | 4 | B | B | B | all |
| 2 | 5 | A | A | A | — (base sample) |
| 2 | 6 | A | B | A | `x2` |
| 2 | 7 | B | B | A | `x1, x2` |
| 2 | 8 | B | B | B | all |

Running the model over every row of `Z` gives a vector `Y`. The Shapley-effect increments
are read straight off consecutive entries of `Y` within each block.

## Two ways to get from samples to results

**Let SAShE drive it** — give it the function and the two sample sets:

```julia
model = CallableModel(f, X1, X2)
Φₙ, Φ²ₙ, Yₙ = analyze(model)          # builds Z, runs f over it, analyses
```

**Drive it yourself** — build the sample, run the model however you like, hand back `Y`:

```julia
S = PickAndFreezeSample(X1, X2)               # Z is S.samples; π is S.permutations
Y = map(row -> f(collect(row)), eachrow(S.samples))
Φₙ, Φ²ₙ = analyze(S, Y)
```

The second form is what you want when the model evaluation needs its own batching,
checkpointing, or a compute cluster — or when the factors are dependent and `Z` needs to be
conditionally resampled (`PickAndFreezeSample(X1, X2; conditional_sampler = ...)`). A dedicated
guide for the dependent case is in [Next steps](@ref).

## What `analyze` returns

`Φₙ` and `Φ²ₙ` are `d × N` matrices of per-sample **increments**, not the final numbers.
Summing a row of `Φₙ` gives that factor's Shapley effect; `SAShE.shapley_effects`,
`SAShE.confint`, and `SAShE.margin_of_error` do that sum and the confidence-interval
arithmetic for you. Keeping the per-sample increments is what makes the unbiased confidence
intervals possible without bootstrapping.
