# `ishigami` and the exact reference values live in test/reference_values.jl, so that both
# this file and the rest of the suite can be run independently of each other.

@testset "Ishigami function" begin
    Random.seed!(0987)

    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 20000, 3
    du = Uniform(-π, π)

    # Shape n_samples ⋅ n_factors
    samples1 = DataFrame(hcat([rand(du, n_factors) for _ ∈ 1:n_samples]...)', factor_names)
    samples2 = DataFrame(hcat([rand(du, n_factors) for _ ∈ 1:n_samples]...)', factor_names)

    m = SAShE.CallableModel(ishigami)
    S = SAShE.CallablePickAndFreezeSample(samples1, samples2)
    Φₙ, Φ²ₙ, Yₙ = SAShE.analyze(m, S)
    Φ, Φlb, Φub = SAShE.shapley_effects(Φₙ, Φ²ₙ)
    Φ_confint = SAShE.confint(Φₙ, Φ²ₙ)
    Φ_moe = SAShE.margin_of_error(Φₙ, Φ²ₙ)

    @test all(Φ_confint[2] .- Φ_confint[1] .== (Φub .- Φlb))
    @test all(2 .* Φ_moe .≈ Φub .- Φlb)

    # Verify the margin-of-error arithmetic against the textbook formula computed a different
    # way: `Φₙ`'s columns are per-sample increments already carrying a 1/N factor, so the
    # per-sample values are N·Φₙ[j, :], and the standard error is their (Bessel-corrected)
    # std over sqrt(N). This replaces an earlier pin of the interval *width* to hardcoded
    # numbers -- a width is itself a random quantity, so pinning it tested nothing about
    # correctness while re-rolling on any change that shifted the RNG stream.
    N = size(Φₙ, 2)
    moe_reference = [1.96 * std(N .* Φₙ[j, :]) / sqrt(N) for j ∈ 1:n_factors]
    @test all(isapprox.(Φ_moe, moe_reference; rtol=1e-10))

    @test all(abs.(Φ .- Φ_ISHIGAMI_EXACT) .< (0.1 .* Φ_ISHIGAMI_EXACT))
end


@testset "Correctness of CallablePickAndFreezeSample assessment" begin
    # Seeded: this testset was previously unseeded, so it drew off whatever global RNG state
    # preceded it and its `sum(Φ) ≈ var(Yₙ)` ratio check could fail by chance at n=1024.
    # That is the flakiness tracked in issue #14.
    Random.seed!(4321)

    factor_names = [:x1, :x2, :x3]
    n_samples = 1024
    n_factors = length(factor_names)

    du = Uniform(-π, π)

    # Shape n_samples ⋅ n_factors
    samples1 = DataFrame(rand(du, n_samples, n_factors), factor_names)
    samples2 = DataFrame(rand(du, n_samples, n_factors), factor_names)

    m = SAShE.CallableModel(ishigami)
    S = SAShE.CallablePickAndFreezeSample(samples1, samples2)

    Φₙ, Φ²ₙ, Yₙ = SAShE.analyze(m, S)
    Φ, Φlb, Φub = SAShE.shapley_effects(Φₙ, Φ²ₙ)

    Φ_confint = SAShE.confint(Φₙ, Φ²ₙ)
    Φ_moe = SAShE.margin_of_error(Φₙ, Φ²ₙ)

    # Shapley effects must sum to the output variance (efficiency). `@test cond || "msg"` was
    # used here before: when `cond` is false that expression evaluates to a `String`, which
    # Test.jl records as an Error rather than a clean Fail, obscuring the diagnostic.
    @test min(var(Yₙ), sum(Φ)) / max(var(Yₙ), sum(Φ)) > 0.95

    # Rebuilding the same S.samples/permutations pair and re-running analyze must reproduce
    # identical results.
    S2 = SAShE.CallablePickAndFreezeSample(S.samples, S.permutations)
    Φₙ2, Φ²ₙ2, _ = SAShE.analyze(m, S2)
    Φ2, Φlb2, Φub2 = SAShE.shapley_effects(Φₙ2, Φ²ₙ2)

    @test all(Φ .== Φ2)
end
