# Examples

One example per combination of "what you have" and "which estimation method" — click a
cell to jump to its section.

**Table 1** (as on the home page). Click a cell for its example or status.

| What you have | Pick-and-freeze | Double Monte Carlo |
| :-- | :-- | :-- |
| Model, known-distribution inputs | [✅ Example](@ref "Model, known distribution — pick-and-freeze") | [✅ Example](@ref "Model, known distribution — double Monte Carlo") |
| Model, real-data inputs ('mix') | [✅ Example](@ref "Model, real data ('mix') — pick-and-freeze") | [✅ Example](@ref "Model, real data ('mix') — double Monte Carlo") |
| No model — real data throughout | [✅ Example](@ref "No model, real data — pick-and-freeze") | [🔲 Planned](@ref "No model, real data — double Monte Carlo") |

## Model, known distribution — pick-and-freeze

`CallableModel` + `CallablePickAndFreezeSample`. Full walk-through: [Getting started](@ref).

```@example ex_model_known_pf
using SAShE, DataFrames, Distributions, Random
Random.seed!(1)

ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

d = Uniform(-π, π)
X1 = DataFrame(rand(d, 2000, 3), [:x1, :x2, :x3])
X2 = DataFrame(rand(d, 2000, 3), [:x1, :x2, :x3])

model = CallableModel(ishigami)
S = CallablePickAndFreezeSample(X1, X2)
Φₙ, Φ²ₙ, Yₙ = analyze(model, S)
SAShE.shapley_effects(Φₙ)
```

Dependent inputs use the same type, with a `conditional_sampler` — see the "Dependent
factors" section of [Next steps](@ref).

## Model, known distribution — double Monte Carlo

`CallableModel` + `CallableDoubleMonteCarloSample`, independent factors only for now.

```@example ex_model_known_dmc
using SAShE, DataFrames, Distributions, Random
Random.seed!(11)

ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

factor_names = [:x1, :x2, :x3]
dists = fill(Uniform(-π, π), 3)

model = CallableModel(ishigami)
S = CallableDoubleMonteCarloSample(factor_names, dists, 3000, MonteCarloSampling())
Φₙ, Φ²ₙ, Yₙ = analyze(model, S)
SAShE.shapley_effects(Φₙ)
```

Unlike pick-and-freeze, this calls the model at genuinely new points — `N_V + m·N_I·N_O·(d-1)`
evaluations, not reused ones. See [How it works](@ref) for why this estimator exists
alongside pick-and-freeze rather than replacing it.

## Model, real data ('mix') — pick-and-freeze

`MixModel`. For when a model is callable but its inputs' joint distribution isn't known well
enough to sample fresh, valid points from — only an existing dataset. A nearest-neighbour
lookup into that dataset picks a real point to build a synthetic evaluation point from,
instead of sampling from a known distribution (`CallableModel`) or reusing stored
outputs (`DataModel`).

```@example ex_mix_pf
using SAShE, DataFrames, Distributions, Random
Random.seed!(3)

ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

d = Uniform(-π, π)
X = DataFrame(rand(d, 5000, 3), [:x1, :x2, :x3])
Y = map(row -> ishigami(collect(row)), eachrow(X))

model = MixModel(ishigami)
S = MixPickAndFreezeSample(X, Y, 5000)
Φₙ, Φ²ₙ, Yₙ = analyze(model, S)
SAShE.shapley_effects(Φₙ)
```

Unlike `DataModel`'s "knn" estimator, `MixModel` calls `func` at every synthetic point it
builds — including, faithfully to [1]'s Eq. (22), the reference point itself, whose output
is already known exactly as `Y[s]`.

## Model, real data ('mix') — double Monte Carlo

`MixModel`, paired with `MixDoubleMonteCarloSample` instead. Its constructor accepts an
`N_I` keyword (default `3`) controlling how many nearest-neighbour points the inner sample
variance is built from.

```@example ex_mix_dmc
using SAShE, DataFrames, Distributions, Random
Random.seed!(4)

ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

d = Uniform(-π, π)
X = DataFrame(rand(d, 5000, 3), [:x1, :x2, :x3])
Y = map(row -> ishigami(collect(row)), eachrow(X))

model = MixModel(ishigami)
S = MixDoubleMonteCarloSample(X, Y, 5000)
Φₙ, Φ²ₙ, Yₙ = analyze(model, S)
SAShE.shapley_effects(Φₙ)
```

## No model, real data — pick-and-freeze

`DataModel`. Full walk-through: [Dataset-only workflow](@ref).

```@example ex_nomodel_data_pf
using SAShE, DataFrames, Distributions, Random
Random.seed!(2)

ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

d = Uniform(-π, π)
X = DataFrame(rand(d, 5000, 3), [:x1, :x2, :x3])
Y = map(row -> ishigami(collect(row)), eachrow(X))

model = DataModel(X, Y)
Φₙ, Φ²ₙ = analyze(model, 5000, PickAndFreeze())
SAShE.shapley_effects(Φₙ)
```

Zero new evaluations — every estimate reuses a `Y` value already present in the dataset.

## No model, real data — double Monte Carlo

🔲 Not yet implemented.
