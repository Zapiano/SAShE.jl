using LinearAlgebra: cholesky

# `_linear_gaussian_theoretical_shapley` comes from test/reference_values.jl.

@testset "Nearest-neighbour Shapley effects: correctness vs. linear-Gaussian closed form ([1] §6.3-style benchmark)" begin
    Random.seed!(22)

    p = 3
    β = [1.0, 1.0, 1.0]
    # x1, x3 correlated (ρ=0.6); x2 independent — mirrors the dependent-input benchmark
    # setup used throughout [1], §5.2/§6.3. See docs/src/references.md for citation [1].
    Γ = [1.0 0.0 0.6
         0.0 1.0 0.0
         0.6 0.0 1.0]

    η_theoretical = _linear_gaussian_theoretical_shapley(β, Γ)

    n_samples = 8000
    Xm = randn(n_samples, p) * cholesky(Γ).U
    X = DataFrame(Xm, [:x1, :x2, :x3])
    Y = collect(Xm * β)

    Φₙ, Φ²ₙ = SAShE.analyze(DataModel(X, Y), 6000, PickAndFreeze())
    Φ = SAShE.shapley_effects(Φₙ)

    @test all(abs.(Φ .- η_theoretical) .< 0.15 .* η_theoretical)
end

@testset "Nearest-neighbour Shapley effects: regression vs. exact estimator (independent Ishigami)" begin
    # Averaged over several independent datasets rather than checking a single seeded run.
    # This estimator's per-run spread is large -- measured std ≈ 0.47 on x3, whose true
    # effect is ≈1.687, i.e. ~28% relative -- so a single-run check against a 0.3 tolerance is
    # roughly a 1σ band and fails for about a third of seeds (5 of 12 measured). Two earlier
    # attempts to stabilise this by picking a luckier seed each broke again as soon as an
    # unrelated change shifted the RNG stream, which is the tell that the test, not the
    # estimator, was at fault. Averaging 5 replicates tightens the tested quantity to a
    # measured worst case of 0.18 max relative error across 6 disjoint seed blocks (vs 0.60
    # for single runs), so the 0.3 tolerance below is now genuine headroom rather than a
    # coin flip, at comparable runtime.
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 4000, length(factor_names)
    n_replicates, n_permutations = 5, 2000
    du = Uniform(-π, π)

    Φ_total = zeros(n_factors)
    for seed ∈ 1:n_replicates
        Random.seed!(seed)
        X = DataFrame(rand(du, n_samples, n_factors), factor_names)
        Y = map(x -> ishigami(collect(x)), eachrow(X))

        Φₙ, _ = SAShE.analyze(DataModel(X, Y), n_permutations, PickAndFreeze())
        Φ_total .+= SAShE.shapley_effects(Φₙ)
    end
    Φ = Φ_total ./ n_replicates

    # Same theoretical baseline already validated (against the exact estimator) in
    # test/ishigami.jl — the nearest-neighbour estimator is expected to be noisier (Pick-and-Freeze has
    # higher variance than double-MC, per [1]'s own findings), hence the looser tolerance
    # than the exact-estimator test uses. See docs/src/references.md for citation [1].
    Φ_base_vals = Φ_ISHIGAMI_EXACT  # see its derivation in test/reference_values.jl
    @test all(abs.(Φ .- Φ_base_vals) .< 0.3 .* Φ_base_vals)
end

@testset "Nearest-neighbour Shapley effects: error path for N_I > N" begin
    X = DataFrame(randn(1, 3), [:x1, :x2, :x3])  # only 1 row: N_I=1 needs ≥2
    Y = collect(sum.(eachrow(X)))

    @test_throws ArgumentError SAShE.analyze(DataModel(X, Y), 10, PickAndFreeze())
end
