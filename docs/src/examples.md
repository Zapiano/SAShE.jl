# Examples

This page provides a comprehensive list of examples comprising all combinations of *data settings* and *estimation methods* (see [Table 1](index.md#table-1)).


```@raw html
<style>
  #examples-table td.example-cell {
    padding: 0;
    transition: background-color 0.1s ease-in-out;
  }
  #examples-table td.example-cell:hover {
    background-color: rgba(128, 128, 128, 0.2);
  }
  #examples-table td.example-cell a {
    display: block;
    box-sizing: border-box;
    width: 100%;
    height: 100%;
    padding: 0.5em 0.75em;
    text-decoration: none;
    color: inherit;
  }
</style>
<table id="examples-table">
  <thead>
    <tr>
      <th rowspan="2" style="vertical-align: bottom;"><em>Data setting</em></th>
      <th colspan="2" align="center"><em>Estimation method</em></th>
    </tr>
    <tr>
      <th align="left"><em>Pick-and-Freeze</em></th>
      <th align="left"><em>Double Monte Carlo</em></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>CallableModel</code></td>
      <td class="example-cell"><a href="#Model,-known-distribution-—-pick-and-freeze"><code>CallableModel</code> using <em>Pick-and-Freeze</em></a></td>
      <td class="example-cell"><a href="#Model,-known-distribution-—-double-Monte-Carlo"><code>CallableModel</code> using <em>Double Monte Carlo</em></a></td>
    </tr>
    <tr>
      <td><code>MixModel</code></td>
      <td class="example-cell"><a href="#Model,-real-data-('mix')-—-pick-and-freeze"><code>MixModel</code> using <em>Pick-and-Freeze</em></a></td>
      <td class="example-cell"><a href="#Model,-real-data-('mix')-—-double-Monte-Carlo"><code>MixModel</code> using <em>Double Monte Carlo</em></a></td>
    </tr>
    <tr>
      <td><code>DataModel</code></td>
      <td class="example-cell"><a href="#No-model,-real-data-—-pick-and-freeze"><code>DataModel</code> using <em>Pick-and-Freeze</em></a></td>
      <td class="example-cell"><a href="#No-model,-real-data-—-double-Monte-Carlo"><code>DataModel</code> using <em>Double Monte Carlo</em> (🔲 not yet implemented)</a></td>
    </tr>
  </tbody>
</table>
```

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
# Already evaluated S.samples yourself (e.g. an external simulator)? Skip `model` entirely:
# Φₙ, Φ²ₙ, Yₙ = analyze(S, Y)   # Y: one value per row of S.samples, in row order
SAShE.shapley_effects(Φₙ)
```

If you already have a single, even-numbered i.i.d. dataset rather than two separate samples, pass it directly — `CallablePickAndFreezeSample` splits it into two halves for you:

```@example ex_model_known_pf_split
using SAShE, DataFrames, Distributions, Random
Random.seed!(1)

ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

d = Uniform(-π, π)
X = DataFrame(rand(d, 4000, 3), [:x1, :x2, :x3])  # one combined, even-numbered dataset

model = CallableModel(ishigami)
S = CallablePickAndFreezeSample(X)
Φₙ, Φ²ₙ, Yₙ = analyze(model, S)
SAShE.shapley_effects(Φₙ)
```

Dependent inputs use the same type, with a `conditional_sampler` — see `CallablePickAndFreezeSample`'s docstring for the expected signature.

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
# Already evaluated S.samples yourself (e.g. an external simulator)? Skip `model` entirely:
# Φₙ, Φ²ₙ, Yₙ = analyze(S, Y)   # Y: one value per row of S.samples, in row order
SAShE.shapley_effects(Φₙ)
```

Unlike pick-and-freeze, this calls the model at genuinely new points — `N_V + m·N_I·N_O·(d-1)` evaluations, not reused ones. See [Concepts](@ref) for why this estimator exists alongside pick-and-freeze rather than replacing it.

## Model, real data ('mix') — pick-and-freeze

`MixModel`. For when a model is callable but its inputs' joint distribution isn't known well enough to sample fresh, valid points from — only an existing dataset. A nearest-neighbour lookup into that dataset picks a real point to build a synthetic evaluation point from, instead of sampling from a known distribution (`CallableModel`) or reusing stored outputs (`DataModel`).

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

Unlike `DataModel`'s "knn" estimator, `MixModel` calls `func` at every synthetic point it builds — including, faithfully to [[1]](@ref References)'s Eq. (22), the reference point itself, whose output is already known exactly as `Y[s]`.

## Model, real data ('mix') — double Monte Carlo

`MixModel`, paired with `MixDoubleMonteCarloSample` instead. Its constructor accepts an `N_I` keyword (default `3`) controlling how many nearest-neighbour points the inner sample variance is built from.

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

`DataModel`.

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
