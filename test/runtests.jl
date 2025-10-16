using Test
using SAShE
using DataFrames
using Distributions
using Random

function ishigami(X::Vector{Float64}; a::Float64=7.0, b::Float64=0.1)
    return (1 + b * X[3]^4) * sin(X[1]) + a * (sin(X[2]))^2
end

@testset "Ishigami function" begin
    Random.seed!(0987)

    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 20000, 3
    du = Uniform(-π, π)

    # Shape n_samples ⋅ n_factors
    samples1 = DataFrame(hcat([rand(du, n_factors) for _ in 1:n_samples]...)', factor_names)
    samples2 = DataFrame(hcat([rand(du, n_factors) for _ in 1:n_samples]...)', factor_names)

    sa_problem = SAShE.Problem(ishigami, samples1, samples2)
    Φₙ, Φ²ₙ, Yₙ = SAShE.solve(sa_problem)
    Φ, Φlb, Φub = SAShE.shapley_effects(Φₙ, Φ²ₙ)
    Φ_confint = SAShE.confint(Φₙ, Φ²ₙ)
    Φ_moe = SAShE.margin_of_error(Φₙ, Φ²ₙ)

    @test all(Φ_confint[2] .- Φ_confint[1] .== (Φub .- Φlb))
    @test all(2 .* Φ_moe .≈ Φub .- Φlb)

    Φ_confint_base_vals = [0.56, 0.34, 00.37]
    Φ_base_vals = [6.17, 6.08, 1.64]

    @test all(abs.(((Φub .- Φlb) .- Φ_confint_base_vals)) .< (0.1 .* Φ_confint_base_vals))
    @test all(abs.(Φ .- Φ_base_vals) .< (0.1 .* Φ_base_vals))
end

@testset "Correctness of SAShESample assessment" begin
    factor_names = [:x1, :x2, :x3]
    n_samples = 1024
    n_factors = length(factor_names)

    du = Uniform(-π, π)

    # Shape n_samples ⋅ n_factors
    samples1 = DataFrame(rand(du, n_samples, n_factors), factor_names)
    samples2 = DataFrame(rand(du, n_samples, n_factors), factor_names)

    sa_problem = SAShE.Problem(ishigami, samples1, samples2)

    Φₙ, Φ²ₙ, Yₙ = SAShE.solve(sa_problem)
    Φ, Φlb, Φub = SAShE.shapley_effects(Φₙ, Φ²ₙ)

    Φ_confint = SAShE.confint(Φₙ, Φ²ₙ)
    Φ_moe = SAShE.margin_of_error(Φₙ, Φ²ₙ)

    # Compare model variance with sum of Shapley Effects
    @test min(var(Yₙ), sum(Φ)) / max(var(Yₙ), sum(Φ)) > 0.95 || "Ishigami did not converge"

    S_x = SAShESample(samples1, samples2, sa_problem.permutations)
    Y = map(x -> ishigami(collect(x)), eachrow(S_x.samples))
    Φₙ, Φ²ₙ = SAShE.analyze(S_x, Y)
    Φ2, Φlb2, Φub2 = SAShE.shapley_effects(Φₙ, Φ²ₙ)

    # Ensure results are identical to initial analysis
    @test all(Φ .== Φ2) || "Results do not match with same permutations"
end
