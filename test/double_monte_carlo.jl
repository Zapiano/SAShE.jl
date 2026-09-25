@testset "Double Monte Carlo: correctness vs. independent-linear closed form" begin
    Random.seed!(52)

    # For Y = sum(β .* X) with independent X_i, the Shapley effect of X_i is exactly
    # β_i^2 * Var(X_i) -- the model is perfectly additive, so there's no interaction to
    # split between factors, and Owen's decomposition collapses to the first-order effect.
    factor_names = [:x1, :x2, :x3]
    β = [1.0, 2.0, 0.5]
    σ = [1.0, 0.5, 2.0]
    dists = Normal.(0.0, σ)
    model(x) = sum(β .* x)
    η_theoretical = (β .* σ) .^ 2

    m = CallableModel(model)
    S = CallableDoubleMonteCarloSample(factor_names, dists, 3000, MonteCarloSampling(); N_V=8000)
    Φₙ, Φ²ₙ, _ = analyze(m, S)
    Φ = SAShE.shapley_effects(Φₙ)

    @test all(abs.(Φ .- η_theoretical) .< 0.15 .* η_theoretical)
end

@testset "Double Monte Carlo: regression vs. exact estimator (independent Ishigami)" begin
    Random.seed!(63)

    factor_names = [:x1, :x2, :x3]
    du = Uniform(-π, π)
    dists = fill(du, 3)

    m = CallableModel(ishigami)
    S = CallableDoubleMonteCarloSample(factor_names, dists, 3000, MonteCarloSampling(); N_V=5000)
    Φₙ, Φ²ₙ, _ = analyze(m, S)
    Φ = SAShE.shapley_effects(Φₙ)

    # Same tolerance style as the nearest-neighbour estimator's equivalent test
    # (test/nearest_neighbour_shapley_effects.jl) -- relative tolerance on the point
    # estimate, not CI containment, since a single-seed CI check is noisier than needed
    # for a regression test.
    Φ_base_vals = Φ_ISHIGAMI_EXACT  # see its derivation in test/reference_values.jl
    @test all(abs.(Φ .- Φ_base_vals) .< 0.3 .* Φ_base_vals)
end

@testset "Double Monte Carlo: error path for N_I < 2" begin
    factor_names = [:x1, :x2, :x3]
    dists = fill(Normal(), 3)
    @test_throws ArgumentError CallableDoubleMonteCarloSample(
        factor_names, dists, 10, MonteCarloSampling(); N_I=1
    )
end

@testset "Double Monte Carlo: wrap constructor rejects a mismatched sample table" begin
    factor_names = [:x1, :x2, :x3]
    dists = fill(Normal(), 3)
    strategy = MonteCarloSampling()
    S = CallableDoubleMonteCarloSample(factor_names, dists, 5, strategy; N_V=100, N_O=1, N_I=2)

    # Wrong number of factor columns in `permutations` relative to `samples`.
    @test_throws ArgumentError CallableDoubleMonteCarloSample(
        S.samples, S.permutations[:, 1:2], S.N_V, S.N_O, S.N_I
    )

    # Right shape, but a row count that doesn't match N_V/N_O/N_I/m.
    truncated = S.samples[1:(end - 1), :]
    @test_throws ArgumentError CallableDoubleMonteCarloSample(
        truncated, S.permutations, S.N_V, S.N_O, S.N_I
    )

    # A consistent table is accepted.
    @test CallableDoubleMonteCarloSample(S.samples, S.permutations, S.N_V, S.N_O, S.N_I) isa
        CallableDoubleMonteCarloSample
end
