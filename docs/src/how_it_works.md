# How it works

You can use SAShE without reading this page. It is here for when you want to know what
`analyze` is doing, or when you need finer control over the sampling — building the sample
table yourself, or handling dependent factors.

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

## Which method should I use?

`analyze` always takes a model (what you have) and, where one applies, a sample (how to
draw or evaluate it) — `analyze(model, sample)`. Which estimator runs is determined by the
*sample's* type: pair a model with a `CallablePickAndFreezeSample` and you get pick-and-freeze; pair
it with a `CallableDoubleMonteCarloSample` and you get double Monte Carlo ([2] §4.1). `MixModel`
follows the same shape, paired with `MixPickAndFreezeSample` or `MixDoubleMonteCarloSample`
instead. `DataModel` is the one exception — it has no separate sampling step, so it takes the
estimator as an explicit third argument instead (no default — you must say which one), since
one container serves both algorithms there. This section lists what is available for each
case; the sections after it describe the mechanics.

### Read off what's available

| Scenario | Can use |
| :-- | :-- |
| No model, real data | ✅ `analyze(DataModel(...), n, PickAndFreeze())` |
| | 🔲 `analyze(DataModel(...), n, DoubleMonteCarlo())` — planned |
| Model, known distribution, **independent** factors | ✅ `CallableModel` + `CallablePickAndFreezeSample` |
| | ✅ `CallableModel` + `CallableDoubleMonteCarloSample` |
| Model, known distribution, **dependent** factors | ✅ `CallableModel` + `CallablePickAndFreezeSample` (with `conditional_sampler`) |
| | 🔲 `CallableModel` + `CallableDoubleMonteCarloSample` — dependent-factor support planned |
| Model, real-data inputs ("mix") | ✅ `MixModel` + `MixPickAndFreezeSample` |
| | ✅ `MixModel` + `MixDoubleMonteCarloSample` |

### What both methods have in common

- **Pick-and-freeze is implemented by two different formulas**, depending on which type you
  use:
  - `DataModel`'s dataset-only estimator (`_nearest_neighbour_pick_freeze`) computes
    `V_u = E[Z₁Z₂] - E(Y)²` as a literal product of the query point's own output and its
    nearest neighbour's, sharing the same held factors — [1]'s Eq. (23). [1] proves this
    converges in probability as the dataset and permutation count grow (Theorem 3), with
    rates in Theorem 4 — the nearest neighbour only *approximates* a draw from the
    conditional distribution, so this is consistency, not finite-sample unbiasedness.
  - `CallableModel`/`CallablePickAndFreezeSample`'s permutation walk instead implements [4]'s
    Algorithm 1, which estimates the increment `τ̄²_{u+j} - τ̄²_u` directly via a
    product-of-differences (`(Yₙ - midpoint) × (difference)`, see
    [src/callable_model.jl](https://github.com/Zapiano/SAShE.jl/blob/main/src/callable_model.jl)).
    [4] proves this estimator (Algorithm 1) is unbiased by a direct linearity-of-expectation
    argument (§3.2).

  Double Monte Carlo's cost function, `c(u) = E[Var[Y | X₋ᵤ]]`, has a nested-sample-variance
  estimator that [2] states is unbiased for any sample size — which is why [2] estimates `c`
  rather than the alternative `c̃(u) = Var[E[Y|Xᵤ]]` (§3.1).
- **Both get confidence intervals the same way, with no bootstrapping needed** —
  `shapley_effects`, `confint`, and `margin_of_error` assume each permutation's contribution
  is drawn independently, which holds for `CallableModel` but not for `DataModel`/`MixModel`
  (see [Deliberate deviations](@ref)):

  ```julia
  Φ, Φlb, Φub = SAShE.shapley_effects(Φₙ, Φ²ₙ)   # works the same for a CallableDoubleMonteCarloSample's
                                                # or a CallablePickAndFreezeSample's (Φₙ, Φ²ₙ)
  ```

- **Both converge at the same `O(1/√m)` rate** in the number of permutations `m`.

### Where each one is the only option

- **No model, real data only** — `DataModel` (pick-and-freeze) is the only estimator
  implemented.
- **Model, known distribution, dependent factors** — `CallablePickAndFreezeSample` with a
  `conditional_sampler`; double Monte Carlo's dependent-factor support is not implemented.

For the case where both apply (callable model, independent factors), see each method's cost
formula above and the papers behind them — [4] for pick-and-freeze, [2] and [1] for double
Monte Carlo.

## Permutations

For `N` base samples SAShE draws an `N × d` matrix `π`, one random factor ordering per row.
`generate_permutations(N, d)` produces it; `CallablePickAndFreezeSample` and `CallableDoubleMonteCarloSample`
do this for you and store it, so the sampling and the analysis always use the same orderings.

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

## Model and sample, always together

`analyze` always takes a model and a sample built independently — a `CallableModel` never
holds sample data itself, and a sample never holds the model. Build both, then hand them to
`analyze` together:

```julia
model = CallableModel(f)
S = CallablePickAndFreezeSample(X1, X2)       # Z is S.samples; π is S.permutations
Φₙ, Φ²ₙ, Yₙ = analyze(model, S)        # runs f over Z (in parallel, via pmap), then analyses
```

Building `S` yourself this way — rather than relying on some all-in-one shortcut — is what
lets you control the sampling directly: a custom permutation scheme by wrapping an
already-built table (`CallablePickAndFreezeSample(samples, perms)`), or dependent factors via
`conditional_sampler` (`CallablePickAndFreezeSample(X1, X2; conditional_sampler = ...)`). A
dedicated guide for the dependent case is in [Next steps](@ref).

## What `analyze` returns

`Φₙ` and `Φ²ₙ` are `d × N` matrices of per-sample **increments**, not the final numbers.
Summing a row of `Φₙ` gives that factor's Shapley effect; `SAShE.shapley_effects`,
`SAShE.confint`, and `SAShE.margin_of_error` do that sum and the confidence-interval
arithmetic for you. Keeping the per-sample increments is what makes the unbiased confidence
intervals possible without bootstrapping.
