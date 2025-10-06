using Test
using SAShE
using DataFrames
using Distributions
using Random
@testset "Ishigami function" begin
    Random.seed!(0987)

    function ishigami(X::Vector{Float64}; a::Float64=7.0, b::Float64=0.1)
        return (1 + b * X[3]^4) * sin(X[1]) + a * (sin(X[2]))^2
    end

    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 20000, 3
    du = Uniform(-π, π)

    # Shape n_samples ⋅ n_factors
    samples1 = DataFrame(hcat([rand(du, n_factors) for _ in 1:n_samples]...)', factor_names)
    samples2 = DataFrame(hcat([rand(du, n_factors) for _ in 1:n_samples]...)', factor_names)

    sa_problem = SAShE.Problem(ishigami, samples1, samples2)
    Φₙ, Φ²ₙ, Yₙ = SAShE.solve(sa_problem)
    Φ, Φlb, Φub = SAShE.shapley_effects(Φₙ, Φ²ₙ)

    # Compute variance of Yₙ
    mean_val = sum(Yₙ) / n_samples
    squared_diffs = (Yₙ .- Ref(mean_val)) .^ 2
    var_Y = sum(squared_diffs) / (n_samples - 1)
    @test (sum(Φₙ) / var_Y) ≈ 1.0 atol = 1e-2

    Φ_confint = SAShE.confint(Φₙ, Φ²ₙ)
    Φ_moe = SAShE.margin_of_error(Φₙ, Φ²ₙ)

    @test all(Φ_confint[2] .- Φ_confint[1] .== (Φub .- Φlb))
    @test all(2 .* Φ_moe .≈ Φub .- Φlb)

    Φ_confint_base_vals = [0.56, 0.34, 00.37]
    Φ_base_vals = [6.17, 6.08, 1.64]

    @test all(abs.(((Φub .- Φlb) .- Φ_confint_base_vals)) .< (0.1 .* Φ_confint_base_vals))
    @test all(abs.(Φ .- Φ_base_vals) .< (0.1 .* Φ_base_vals))
end
