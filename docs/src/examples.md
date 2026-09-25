# Examples

One example per combination of "what you have" and "which estimation method" — click a
cell to jump to its section.

**Table 1** (as on the home page). Click a cell for its example or status.

| What you have | Pick-and-freeze | Double Monte Carlo |
| :-- | :-- | :-- |
| Model, known-distribution inputs | [✅ Example](@ref "Model, known distribution — pick-and-freeze") | [✅ Example](@ref "Model, known distribution — double Monte Carlo") |
| Model, real-data inputs ('mix') | [🔲 Planned](@ref "Model, real data ('mix') — pick-and-freeze") | [🔲 Planned](@ref "Model, real data ('mix') — double Monte Carlo") |
| No model — real data throughout | [✅ Example](@ref "No model, real data — pick-and-freeze") | [🔲 Planned](@ref "No model, real data — double Monte Carlo") |

## Model, known distribution — pick-and-freeze

`CallableModel` + `PickAndFreezeSample`. Full walk-through: [Getting started](@ref).

```@example ex_model_known_pf
using SAShE, DataFrames, Distributions, Random
Random.seed!(1)

ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

d = Uniform(-π, π)
X1 = DataFrame(rand(d, 2000, 3), [:x1, :x2, :x3])
X2 = DataFrame(rand(d, 2000, 3), [:x1, :x2, :x3])

model = CallableModel(ishigami)
S = PickAndFreezeSample(X1, X2)
Φₙ, Φ²ₙ, Yₙ = analyze(model, S)
SAShE.shapley_effects(Φₙ)
```

Dependent inputs use the same type, with a `conditional_sampler` — see the "Dependent
factors" section of [Next steps](@ref).

## Model, known distribution — double Monte Carlo

`CallableModel` + `DoubleMonteCarloSample`, independent factors only for now.

```@example ex_model_known_dmc
using SAShE, DataFrames, Distributions, Random
Random.seed!(11)

ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

factor_names = [:x1, :x2, :x3]
dists = fill(Uniform(-π, π), 3)

model = CallableModel(ishigami)
S = DoubleMonteCarloSample(factor_names, dists, 3000, MonteCarloSampling())
Φₙ, Φ²ₙ, Yₙ = analyze(model, S)
SAShE.shapley_effects(Φₙ)
```

Unlike pick-and-freeze, this calls the model at genuinely new points — `N_V + m·N_I·N_O·(d-1)`
evaluations, not reused ones. See [How it works](@ref) for why this estimator exists
alongside pick-and-freeze rather than replacing it.

## Model, real data ('mix') — pick-and-freeze

🔲 Not yet implemented. For when a model is callable but its inputs' joint distribution
isn't known well enough to sample fresh, valid points from — a nearest-neighbour lookup
into an existing dataset would pick a real point to evaluate the model at instead.

## Model, real data ('mix') — double Monte Carlo

🔲 Not yet implemented.

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
