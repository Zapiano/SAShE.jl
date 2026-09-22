# SAShE.jl

Sensitivity analysis with **Shapley effects** — a variance-based measure of how much each
input factor contributes to the variability of a model's output.

Unlike the classical Sobol' main and total effects, Shapley effects **sum exactly to the
total output variance**. Each one therefore reads directly as "this fraction of the
output's variance is attributable to this factor", which makes them easy to compare and to
communicate.

SAShE.jl implements the simple Monte Carlo estimator of [4] — see [References](@ref) — it
estimates every factor's Shapley effect simultaneously at a cost of `N·(d + 1)` model
evaluations (`N` samples, `d` factors), together with unbiased confidence intervals.

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

model = SAShEModel(ishigami, X1, X2)
Φₙ, Φ²ₙ, Yₙ = analyze(model)

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

See the [References](@ref) page for full citations.
