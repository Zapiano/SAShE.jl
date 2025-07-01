# SAShE.jl

This package performs a Sensitivity Analysis using Shapley Effects given a model `my_func` that accepts a vector of factors `X`. The approach implemented here was presented in [1]. If the user is using `Distributed` and has added some procs with `addprocs` the solution will be run in multiple cores.

The "examples/ishigami.jl" script can be used to compare the result of this implementation with the one from the paper for the Ishigami function.

# Quick start

Assuming a function `my_func` that  accepts a vector of factors`X`

```
my_func(X::Vector{Float64}) = X[1] + X[2]^2 + X[3]^3
```

First create two separate sample DataFrames with the same shape:

```
using DataFrames

n_factors = 3
n_samples = 1000

X1 = DataFrame(rand(n_samples, n_factors), :auto)
X2 = DataFrame(rand(n_samples, n_factors), :auto)
```

Then create a `SAShE.Problem` instance and solve it:

```
using SAShE

sa_problem = SAShE.Problem(my_func, X1, X2)
Φₙ, Φₙ_var, Yₙ = SAShE.solve(sa_problem)

# Vector of Shapley Effects for each factor
Φ = sum(Φₙ, dims=2)
```

If/when the analysis converges, the sum of the Shapley Effects of all factors should approach the total variance of the model.

```
using Statistics

min(var(Yₙ), sum(Φ))/max(var(Yₙ), sum(Φ)) > 0.95
```

# Reference

1. Goda, T. (2021). A simple algorithm for global sensitivity analysis with Shapley effects. Reliability Engineering & System Safety, 213, 107702.
