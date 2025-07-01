# Example with the ishigami function, the first from the paper
# Also shows how to use Distributed to

Distributed.addprocs(5)     # replace by a reasonable number.

@everywhere begin
    using SAShE
    using DataFrames
    using Random
    using Distributions
    using Distributed

    Random.seed!(0987)

    function ishigami(X::Vector{Float64})
        a::Float64 = 7.0
        b::Float64 = 0.1
        return (1 + b * X[3]^4) * sin(X[1]) + a * (sin(X[2]))^2
    end

    factor_names = [:x1, :x2, :x3]
    n_samples = 10000
    n_factors = 3                   # x1, x2 and x3

    du = Uniform(-π, π)

    # Shape n_samples ⋅ n_factors
    samples1 = DataFrame(hcat([rand(du, n_factors) for _ in 1:n_samples]...)', factor_names)
    samples2 = DataFrame(hcat([rand(du, n_factors) for _ in 1:n_samples]...)', factor_names)

    sa_problem = SAShE.Problem(ishigami, samples1, samples2)
end

Φₙ, Φₙ_var, Yₙ = SAShE.solve(sa_problem)

Φ = sum(Φₙ, dims=2)

var(Yₙ)
sum(Φ)

min(var(Yₙ), sum(Φ)) / max(var(Yₙ), sum(Φ)) > 0.95
