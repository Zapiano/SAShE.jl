# Getting started

Variance based sensitivity analysis asks: **which input factors actually drive the variability of a
model's output, and by how much?** ([5] — see [References](@ref)) That matters for deciding which
inputs are worth measuring more precisely, which ones can be fixed at a nominal value with little
loss of accuracy, and where a model's behaviour is being driven by an interaction between factors
rather than any one of them alone.

**Shapley effects** answer this with a single number per factor, borrowed from
cooperative game theory: treat each factor as a "player" contributing to the "payoff"
(the output's variance), and split that payoff fairly among the players based on their
average marginal contribution across every possible order in which they could be
revealed. Two properties make this attractive over the classical Sobol' main/total
effects:

- **They sum exactly to the total output variance** — nothing is double-counted between
  interacting factors, and nothing is left unattributed.
- **They handle interactions and dependent factors** without needing a separate
  decomposition for each — a factor with zero main effect but a strong interaction
  still gets a non-zero share.

The trade-off is cost: computing them exactly means evaluating every possible subset of
factors, which is intractable beyond a handful of factors. SAShE.jl implements
Monte Carlo estimators of the Shapley effects that avoid that blow-up (see
[How it works](@ref)).

## The running example

This page walks through a full analysis of the
[Ishigami function](https://en.wikipedia.org/wiki/Ishigami_function), a standard
sensitivity-analysis test case with three independent inputs on `[-π, π]`.

## 1. Define the model

The model is any function that takes a `Vector` of factor values and returns a scalar:

```@example gs
using SAShE, DataFrames, Distributions, Random
Random.seed!(0987)

function ishigami(x)
    a, b = 7.0, 0.1
    return (1 + b * x[3]^4) * sin(x[1]) + a * sin(x[2])^2
end
nothing # hide
```

## 2. Draw two sample sets

SAShE needs **two** independent sample sets of the same shape, `X1` and `X2` (called `x`
and `y` in [4] — see [References](@ref)). Each row is one draw of all factors; each column
is a factor.

```@example gs
d = Uniform(-π, π)
n_samples, n_factors = 10000, 3
factor_names = [:x1, :x2, :x3]

X1 = DataFrame(rand(d, n_samples, n_factors), factor_names)
X2 = DataFrame(rand(d, n_samples, n_factors), factor_names)
nothing # hide
```

Here every factor is independent and uniform. If your factors have different marginal
distributions, draw each column from its own. If the factors are *dependent* (for example
one is a function of the others), the sampling needs an extra step — see [Next steps](@ref).

## 3. Run the analysis

Wrap the model in a `CallableModel`, build a `PickAndFreezeSample` from your two sample
sets, then `analyze` them together — `analyze` always takes a model and a sample:

```@example gs
model = CallableModel(ishigami)
S = PickAndFreezeSample(X1, X2)
Φₙ, Φ²ₙ, Yₙ = analyze(model, S)
nothing # hide
```

`analyze` returns three objects:

| Return value | Shape  | Meaning |
|:-------------|:-------|:--------|
| `Φₙ`         | `d × N` | per-sample contributions to each factor's Shapley effect |
| `Φ²ₙ`        | `d × N` | per-sample contributions used for the confidence intervals |
| `Yₙ`         | `N`     | the model output at each `X1` row |

These are **increments**, not the final numbers — see step 4.

## 4. Turn the increments into estimates

```@example gs
Φ = SAShE.shapley_effects(Φₙ)
```

With `Φ²ₙ` as a second argument you also get the 95% confidence bounds:

```@example gs
Φ, Φlb, Φub = SAShE.shapley_effects(Φₙ, Φ²ₙ)
[Φlb Φ Φub]
```

or the pieces on their own:

```@example gs
SAShE.margin_of_error(Φₙ, Φ²ₙ)   # half-width of each interval
```

```@example gs
SAShE.confint(Φₙ, Φ²ₙ)           # (lower, upper)
```

## 5. Sanity check: do they sum to the variance?

The Shapley effects sum to the total output variance once the estimator has converged:

```@example gs
sum(Φ), var(Yₙ)
```

If these disagree by more than a few percent, increase `n_samples`.

## Interpreting the result

The exact Shapley effects for the Ishigami function with `a = 7`, `b = 0.1` and
`xᵢ ~ U(-π, π)` are `[6.0327, 6.1250, 1.6869]` for `x1`, `x2`, `x3`, summing to the total
output variance `13.8446`. The estimates above are a finite-sample approximation of these.

These come from the function's ANOVA decomposition, which has only three non-zero components:

```math
\\sigma^2_1 = \\tfrac{1}{2}\\left(1 + \\tfrac{b\\pi^4}{5}\\right)^2, \\qquad
\\sigma^2_2 = \\tfrac{a^2}{8}, \\qquad
\\sigma^2_{13} = b^2\\pi^8\\left(\\tfrac{1}{18} - \\tfrac{1}{50}\\right)
```

with `σ²₃ = σ²₁₂ = σ²₂₃ = σ²₁₂₃ = 0`. Owen's decomposition, `φᵢ = Σ_{u ∋ i} σ²_u / |u|`, then
gives `φ₁ = σ²₁ + σ²₁₃/2`, `φ₂ = σ²₂`, `φ₃ = σ²₁₃/2`. The same values and derivation are used
as the reference throughout the test suite (`test/reference_values.jl`).

- `x1` and `x2` contribute almost equally.
- `x3` has **zero main effect** — on its own it explains none of the variance — yet its
  Shapley effect is clearly non-zero, because it interacts with `x1`. This is exactly the
  kind of structure Shapley effects surface that a main-effect analysis would miss.
