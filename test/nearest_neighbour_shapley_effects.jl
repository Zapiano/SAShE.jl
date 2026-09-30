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

@testset "DataModel + DoubleMonteCarlo: correctness vs. independent-linear closed form" begin
    # Same averaging rationale as the equivalent MixModel test in test/mix_model.jl -- per-run
    # spread from a single seed is comparable to the smallest effect being measured.
    β = [1.0, 2.0, 0.5]
    σ = [1.0, 0.5, 2.0]
    dists = Normal.(0.0, σ)
    model(x) = sum(β .* x)
    η_theoretical = (β .* σ) .^ 2
    n_samples, n_replicates, n_permutations = 8000, 5, 4000

    Φ_total = zeros(3)
    for seed ∈ 1:n_replicates
        Random.seed!(70 + seed)
        X = DataFrame(hcat(rand.(dists, n_samples)...), [:x1, :x2, :x3])
        Y = map(row -> model(collect(row)), eachrow(X))
        Φₙ, _ = SAShE.analyze(DataModel(X, Y), n_permutations, DoubleMonteCarlo())
        Φ_total .+= SAShE.shapley_effects(Φₙ)
    end
    Φ = Φ_total ./ n_replicates

    @test all(abs.(Φ .- η_theoretical) .< 0.15 .* η_theoretical)
end

@testset "DataModel + DoubleMonteCarlo: correctness vs. dependent-linear-Gaussian closed form" begin
    p = 3
    β = [1.0, 1.0, 1.0]
    # x1, x3 correlated (ρ=0.6); x2 independent — same benchmark shape as the PickAndFreeze
    # test above.
    Γ = [1.0 0.0 0.6
         0.0 1.0 0.0
         0.6 0.0 1.0]
    η_theoretical = _linear_gaussian_theoretical_shapley(β, Γ)
    n_samples, n_permutations = 8000, 6000

    Random.seed!(23)
    Xm = randn(n_samples, p) * cholesky(Γ).U
    X = DataFrame(Xm, [:x1, :x2, :x3])
    Y = collect(Xm * β)

    Φₙ, _ = SAShE.analyze(DataModel(X, Y), n_permutations, DoubleMonteCarlo())
    Φ = SAShE.shapley_effects(Φₙ)

    @test all(abs.(Φ .- η_theoretical) .< 0.15 .* η_theoretical)
end

@testset "DataModel + DoubleMonteCarlo: regression vs. exact estimator (independent Ishigami)" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 4000, length(factor_names)
    n_replicates, n_permutations = 5, 2000
    du = Uniform(-π, π)

    Φ_total = zeros(n_factors)
    for seed ∈ 1:n_replicates
        Random.seed!(seed)
        X = DataFrame(rand(du, n_samples, n_factors), factor_names)
        Y = map(x -> ishigami(collect(x)), eachrow(X))

        Φₙ, _ = SAShE.analyze(DataModel(X, Y), n_permutations, DoubleMonteCarlo())
        Φ_total .+= SAShE.shapley_effects(Φₙ)
    end
    Φ = Φ_total ./ n_replicates

    @test all(abs.(Φ .- Φ_ISHIGAMI_EXACT) .< 0.3 .* Φ_ISHIGAMI_EXACT)
end

@testset "DataModel + DoubleMonteCarlo: N_I must be ≥ 2" begin
    X = DataFrame(randn(10, 3), [:x1, :x2, :x3])
    Y = collect(sum.(eachrow(X)))

    @test_throws ArgumentError SAShE.analyze(
        DataModel(X, Y), 10, DoubleMonteCarlo(); N_I=1
    )
end

@testset "DataModel + DoubleMonteCarlo: analyze is deterministic given a seeded rng" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 200, length(factor_names)
    du = Uniform(-π, π)

    X = DataFrame(rand(du, n_samples, n_factors), factor_names)
    Y = map(row -> ishigami(collect(row)), eachrow(X))
    dm = DataModel(X, Y)

    Φₙ_a, _ = SAShE.analyze(dm, 300, DoubleMonteCarlo(); rng=Xoshiro(11))
    Φₙ_b, _ = SAShE.analyze(dm, 300, DoubleMonteCarlo(); rng=Xoshiro(11))
    @test Φₙ_a == Φₙ_b
end

@testset "_nearest_neighbour_double_monte_carlo: exact values on a hand-built, tie-free dataset" begin
    # Same dataset as `_nearest_neighbour_mix_double_monte_carlo`'s hand-built test in
    # test/mix_model.jl, chosen so the nearest-neighbour ordering restricted to `notU = [2, 3]`
    # is unambiguous (no exact ties).
    X = DataFrame(
        [0.0 0.0 0.0
         1.0 5.0 9.0
         2.0 6.0 1.0
         3.0 1.0 8.0
         4.0 9.0 2.0],
        [:x1, :x2, :x3],
    )
    Xm = Matrix(X)
    func(x) = x[1] + 2 * x[2] + 3 * x[3]  # same linear function as the mix-model test
    Y = map(row -> func(collect(row)), eachrow(X))
    # Row 1: 0.  Row 2: 1+10+27=38.  Row 3: 2+12+3=17.  Row 4: 3+2+24=29.  Row 5: 4+18+6=28.
    @test Y == [0.0, 38.0, 17.0, 29.0, 28.0]
    s = 1

    # u = [1], N_I = 3: nearest 2 external neighbours by columns [2, 3] alone (standardized
    # Euclidean) are row 3 then row 4 (hand-verified against row 2 and row 5, matching
    # test/mix_model.jl's identical neighbour-search test).
    W = SAShE._nearest_neighbour_double_monte_carlo(Xm, Y, s, [1], 3)
    @test W ≈ var([Y[1], Y[3], Y[4]])
    @test W ≈ var([0.0, 17.0, 29.0])

    # Same setup, N_I = 4: exercises N_I at a non-default value through the actual neighbour
    # search, not just the N_I < 2 rejection path -- the third external neighbour (after rows
    # 3, 4) is row 5, with row 2 farthest (same hand-verified distance ordering as above).
    W4 = SAShE._nearest_neighbour_double_monte_carlo(Xm, Y, s, [1], 4)
    @test W4 ≈ var([Y[1], Y[3], Y[4], Y[5]])
    @test W4 ≈ var([0.0, 17.0, 29.0, 28.0])
end
