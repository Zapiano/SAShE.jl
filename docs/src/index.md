# SAShE.jl

Variance-based sensitivity analysis measures how much each input factor contributes to the
variability of a model's output.

SAShE.jl provides a range of methods for this using **Shapley effects**, covering three
cases:

1. The output is model-derived, and the inputs are sampled from some known distribution
2. [TO DO] The output is model-derived, but the inputs are real data with unknown probability distributions
3. Both the output and input are real data with unknown probability distributions

Unlike the classical Sobol' main and total effects, Shapley effects **sum exactly to the
total output variance**. Each one therefore reads directly as "this fraction of the
output's variance is attributable to this factor", which makes them easy to compare and to
communicate.

**Table 1.** Implementation status by use case and estimation method.

```@raw html
<table>
  <thead>
    <tr>
      <th>What you have</th>
      <th colspan="2" align="center">Your options</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td> </td>
      <td>Pick-and-freeze</td>
      <td>Double Monte Carlo</td>
    </tr>
    <tr>
      <td>Model, inputs sampled from a known distribution (independent or dependent)</td>
      <td><code>CallableModel</code> + <code>PickAndFreezeSample</code></td>
      <td><code>CallableModel</code> + <code>DoubleMonteCarloSample</code> (independent factors only)</td>
    </tr>
    <tr>
      <td>Model, inputs are real data with an unknown distribution ("mix")</td>
      <td>🔲 Planned</td>
      <td>🔲 Planned</td>
    </tr>
    <tr>
      <td>No model — both inputs and output are real data</td>
      <td><code>DataModel</code> (<code>analyze(model, n, PickAndFreeze())</code>)</td>
      <td>🔲 Planned</td>
    </tr>
  </tbody>
</table>
```

Every case runs through `analyze(model, sample)` — a model describing what you have, and a
sample describing how to draw or evaluate it — except `DataModel`, which has no separate
sampling step and takes the estimator as an explicit third argument instead (no default),
since one container serves both algorithms there. `CallableModel` + `PickAndFreezeSample` is
[4]'s simple Monte Carlo
estimator — see [References](@ref) — it estimates every factor's Shapley effect
simultaneously at a cost of `N·(d + 1)` model evaluations (`N` samples, `d` factors),
together with unbiased confidence intervals; see [How it works](@ref). `DataModel`'s
pick-and-freeze estimator reuses values already present in the dataset instead — zero new
evaluations; see [Dataset-only workflow](@ref). `CallableModel` + `DoubleMonteCarloSample`
is [2]'s nested-sampling estimator — genuinely new model evaluations, at a different cost
from pick-and-freeze; see [How it works](@ref) and the [Examples](@ref) page for a worked
comparison.

## Installation

```julia
using Pkg
Pkg.add(; url="https://github.com/Zapiano/SAShE.jl")
```

## 30-second example

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
S = PickAndFreezeSample(X1, X2)
Φₙ, Φ²ₙ, Yₙ = analyze(model, S)

Φ, Φlb, Φub = SAShE.shapley_effects(Φₙ, Φ²ₙ)
Φ
```

`Φ[i]` is the Shapley effect of factor `i`; `Φlb[i]` and `Φub[i]` bracket its 95%
confidence interval. See [Getting started](@ref) for a full walk-through and
[How it works](@ref) for the algorithm and the sampling control it gives you.

## Multi-core

`analyze` distributes the model evaluations through `pmap`. Add worker processes before
calling it and nothing else changes:

```julia
using Distributed
addprocs(4)
@everywhere using SAShE
```

See the [References](@ref) page for full citations, and
[Deliberate deviations](@ref) for the places where this package intentionally differs from
the papers it implements.
