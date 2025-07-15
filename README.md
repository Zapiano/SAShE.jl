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
Φ, Φ², Yₙ = SAShE.solve(sa_problem)
```

The three objects returned are:

- `Φₙ` : The contribution that each sample gives to the each factor's Shapley Effect expected value. The Shapley Effect for each factor, `Φ`, can be calculated by summing all columns of each row of `Φₙ`;
- `Φ²ₙ` : The contribution that each sample gives to each Shapley Effect squared expected valued (`E[Φ²]`). This can be used to calculate the confidence intervals (see below);
- `Yₙ` : The value of the model calculated for each sample on `X1`. Besides being used in the Shapley Effects computation, the variance of `Yₙ` can be compared to the sum of the estimated Shapley Effects `Φ`. If the algorithm has converged, the sum of all Shapley Effects should approach the model's variance.

## Shapley effects and confidence intervals

The function `shapley_effects` returns each factor's Shapley Effect. If used with the second argument `Φ²ₙ`, it also returns the lower and upper bounds of each factor's Shapley Effect confidence interval.

```
# Vector of Shapley Effects for each factor
Φ = SAShE.shapley_effects(Φₙ)

# Or with confidence intervals

Φ, Φ_lb , Φ_ub = SAShE.shapley_effects(Φₙ, Φ²ₙ)
```

The confidence intervals and margin of errors can also be accessed directly via:

```
# Margin of error
SAShE.margin_of_error(Φₙ, Φ²ₙ)

# Confidence interval
SAShE.confint(Φₙ, Φ²ₙ)
```

# Reference

1. Goda, T. (2021). A simple algorithm for global sensitivity analysis with Shapley effects. Reliability Engineering & System Safety, 213, 107702.
