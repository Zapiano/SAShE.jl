# SAShE.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16777876.svg)](https://doi.org/10.5281/zenodo.16777876) [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle) [![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://Zapiano.github.io/SAShE.jl)

Sensitivity analysis with **Shapley effects** — a variance-based measure of how much each
input factor contributes to the variability of a model's output, that sums exactly to the
total output variance.

**Full documentation: [Zapiano.github.io/SAShE.jl](https://Zapiano.github.io/SAShE.jl)** —
start with [Getting started](https://Zapiano.github.io/SAShE.jl/getting_started/) for a
full walk-through, or [How it works](https://Zapiano.github.io/SAShE.jl/how_it_works/) for
the algorithm.

## Installation

```julia
using Pkg
Pkg.add(; url="https://github.com/Zapiano/SAShE.jl")
```

## Quick start

```julia
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
Φ
```

`Φ[i]` is the Shapley effect of factor `i`; `Φlb[i]` and `Φub[i]` bracket its 95%
confidence interval.

## Contributing

This project uses [BlueStyle](https://domluna.github.io/JuliaFormatter.jl/dev/blue_style/) style guide with some extra configuration. If you use VSCode, you don't need to install any extensions, the `.JuliaFormatter.toml` will be automatically be used to format the files.

## References

See the [References](https://Zapiano.github.io/SAShE.jl/references/) page in the docs for
citations.
