# SAShE.jl

SAShE.jl is a package form performing variance-based *Sensitivity analysis with Shapley effects*. Variance-based sensitivity analysis measures how much each input factor contributes to the variability of an output, whether that output comes from evaluating a model, or is itself observed data. One way of performing Variance-based sensitivity analysis is through the computation of Sobol' main and total effects. A more recent approach leverages the concept of Shapley values to calculate indexes based on Sobol's main effect, but with the property that they **sum exactly to the total output variance**. The result therefore reads directly as "this fraction of the output's variance is attributable to this factor", which makes them easy to interpret, compare and communicate. These indexes have been called **Shapley effects**. See [Concepts](@ref) for more details.

SAShE.jl provides two *estimation methods* to estimate **Shapley effects**, Pick and Freeze and Double Monte Carlo, covering three following *data settings*:

1. `CallableModel`: The output is model-derived, and the inputs are sampled from some known distribution
2. `MixModel`: The output is model-derived, but the inputs are "real data" with unknown probability distributions
3. `DataModel`: Both the output and input are "real data" with unknown probability distributions

The first two *data settings* run through `analyze(model, sample)`, where the model (`CallableModel` or `MixModel`) describes what you have, and the sample (`CallablePickAndFreezeSample`, `CallableDoubleMonteCarloSample`, `MixPickAndFreezeSample` and `MixDoubleMonteCarloSample`) supplies the data to evaluate it over, along with the pre-drawn randomness the run needs. The sample's *type* is what selects the estimator, so these two *data settings* take no estimator argument. The third *data setting* (`DataModel`) has no separate sampling step and takes the estimator as an explicit third argument instead (no default). Each *estimation method* is implemented separately for each *data setting*, without loss of generality, to account for the differences between them. See [Estimators](@ref) for more details.


**Table 1.** What to pass, which paper implements it, and what it costs for each *data setting* (rows) and *estimation method* (columns).

```@raw html
<table id="table-1">
  <thead>
    <tr>
      <th rowspan="2" style="vertical-align: bottom;"><em>Data setting</em></th>
      <th colspan="2" align="center"><em>Estimation method</em></th>
    </tr>
    <tr>
      <th align="left">Pick-and-freeze</th>
      <th align="left">Double Monte Carlo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>1.</b> Model, inputs from a known distribution<br>(independent or dependent factors)</td>
      <td>✅ <code>CallableModel</code> + <code>CallablePickAndFreezeSample</code><br>
          <a href="references/#References">[4]</a>, §3.1 Algorithm 1<br>
          Walks one random permutation per base sample, reading every factor's increment off
          consecutive evaluations — <code>N·(d + 1)</code> model calls, with unbiased
          confidence intervals.</td>
      <td>✅ <code>CallableModel</code> + <code>CallableDoubleMonteCarloSample</code><br>
          <a href="references/#References">[2]</a>, §4.1 Algorithm 1<br>
          Estimates each coalition's cost by nested inner/outer sampling at genuinely new
          points — <code>N_V + m·N_I·N_O·(d − 1)</code> model calls. Independent factors
          only.</td>
    </tr>
    <tr>
      <td><b>2.</b> Model, inputs are real data with an unknown distribution ("mix")</td>
      <td>✅ <code>MixModel</code> + <code>MixPickAndFreezeSample</code><br>
          <a href="references/#References">[1]</a>, §6.1.2 Eq. (22)<br>
          Calls the model at a synthetic point pairing the reference row's held factors with
          its nearest neighbour's — <code>2·m·(d − 1)</code> model calls.</td>
      <td>✅ <code>MixModel</code> + <code>MixDoubleMonteCarloSample</code><br>
          <a href="references/#References">[1]</a>, §6.1.1 Eq. (17)<br>
          Inner variance over <code>N_I</code> synthetic points that hold the reference row's
          <em>un</em>-held factors fixed — <code>N_I·m·(d − 1)</code> model calls.</td>
    </tr>
    <tr>
      <td><b>3.</b> No model — inputs and output both real data</td>
      <td>✅ <code>analyze(DataModel(X, Y), m, PickAndFreeze())</code><br>
          <a href="references/#References">[1]</a>, §6.1.2 Eq. (23)<br>
          Multiplies the reference row's own output by its nearest neighbour's, both already
          in the dataset — zero model calls.</td>
      <td>🔲 Planned<br>
          <a href="references/#References">[1]</a>, §6.1.1 Eq. (18)<br>
          The "knn" inner variance, reusing dataset outputs rather than evaluating — would
          also cost zero model calls.</td>
    </tr>
  </tbody>
</table>
```

Every cell above combines its estimator with the same random-permutation W-aggregation procedure ([[2]](@ref References)'s Eq. 12, in the form [[1]](@ref References) states it in §4.2, Eq. 14), and returns increments in the same shape, so `SAShE.shapley_effects` and friends work unchanged throughout. `m` is the number of permutations, `d` the number of factors, and `N` the number of base samples.



## Installation

```julia
using Pkg
Pkg.add(; url="https://github.com/Zapiano/SAShE.jl")
```

## 30-second example

The example below uses Pick-and-Freeze to estimate Shapley effects for the *data setting* 1 (`CallableModel`). See [Examples](@ref) for other examples.

```@example quickstart
using SAShE, DataFrames, Distributions, Random
Random.seed!(1)

# A model: any function taking a vector of factor values and returning a scalar
ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

# Two independent sample sets of the same shape (rows = samples, columns = factors)
d = Uniform(-π, π)
X1 = DataFrame(rand(d, 2000, 3), [:x1, :x2, :x3])
X2 = DataFrame(rand(d, 2000, 3), [:x1, :x2, :x3])

model = CallableModel(ishigami)
S = CallablePickAndFreezeSample(X1, X2)
Φₙ, Φ²ₙ, Yₙ = analyze(model, S)

Φ, Φlb, Φub = SAShE.shapley_effects(Φₙ, Φ²ₙ)
```

`Φ[i]` is the Shapley effect of factor `i`; `Φlb[i]` and `Φub[i]` bracket its 95% confidence interval. See [Getting started](@ref) for a full walk-through.

## Multi-core

`analyze` distributes the model evaluations through `pmap`. Add worker processes before calling it and nothing else changes:

```julia
using Distributed
addprocs(4)
@everywhere using SAShE
```

See [Deliberate deviations](@ref) for the places where this package intentionally differs from the papers it implements.
