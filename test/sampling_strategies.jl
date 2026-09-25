@testset "_draw_factor_blocks: QuasiMonteCarloSampling returns correctly-sized distinct blocks" begin
    # Every block comes from one combined point set, sliced -- so blocks must be distinct
    # and correctly shaped. This matters most for CallableDoubleMonteCarloSample, which needs
    # distinct outer/inner replicates per coalition step.
    #
    # NOTE: distinct blocks alone are *not* evidence the estimator is sound. Deterministic
    # low-discrepancy sequences produce blocks that differ somewhere yet are identical in
    # coordinate 1 (measured cor = 1.0000 for Sobol and Halton), which silently breaks both
    # estimators -- which is why QuasiMonteCarloSampling refuses to wrap them at all. See
    # the rejection testset below.
    du = Uniform(-π, π)
    dn = Normal(0.0, 1.0)
    strategy = QuasiMonteCarloSampling(QMC.LatinHypercubeSample())

    A, B = SAShE._draw_factor_blocks(strategy, [du, dn], [50, 50])

    @test size(A) == (50, 2)
    @test size(B) == (50, 2)
    @test A != B
end

@testset "_draw_factor_blocks: MonteCarloSampling respects each factor's distribution" begin
    Random.seed!(101)
    du = Uniform(-π, π)
    dn = Normal(3.0, 2.0)

    blocks = SAShE._draw_factor_blocks(MonteCarloSampling(), [du, dn], [20000]; rng=Xoshiro(1))
    X = only(blocks)

    @test size(X) == (20000, 2)
    @test all(-π .<= X[:, 1] .<= π)
    @test abs(mean(X[:, 2]) - 3.0) < 0.1
    @test abs(std(X[:, 2]) - 2.0) < 0.1
end

@testset "CallablePickAndFreezeSample(factor_names, n_samples, factor_dist, strategy): drawn fresh, MonteCarloSampling" begin
    Random.seed!(81)

    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 20000, length(factor_names)
    du = Uniform(-π, π)

    S = CallablePickAndFreezeSample(factor_names, n_samples, fill(du, n_factors), MonteCarloSampling())

    @test size(S.samples) == (n_samples * (n_factors + 1), n_factors)
    @test size(S.permutations) == (n_samples, n_factors)

    m = CallableModel(ishigami)
    Φₙ, Φ²ₙ, Yₙ = analyze(m, S)
    Φ = SAShE.shapley_effects(Φₙ)

    Φ_base_vals = Φ_ISHIGAMI_EXACT  # see its derivation in test/reference_values.jl
    @test all(abs.(Φ .- Φ_base_vals) .< (0.1 .* Φ_base_vals))
end

@testset "CallablePickAndFreezeSample(factor_names, n_samples, factor_dist, strategy): drawn fresh, QuasiMonteCarloSampling" begin
    Random.seed!(82)

    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 20000, length(factor_names)
    du = Uniform(-π, π)
    strategy = QuasiMonteCarloSampling(QMC.LatinHypercubeSample())

    S = CallablePickAndFreezeSample(factor_names, n_samples, fill(du, n_factors), strategy)

    @test size(S.samples) == (n_samples * (n_factors + 1), n_factors)
    @test size(S.permutations) == (n_samples, n_factors)

    m = CallableModel(ishigami)
    Φₙ, Φ²ₙ, Yₙ = analyze(m, S)
    Φ = SAShE.shapley_effects(Φₙ)

    # No unbiasedness guarantee under QMC-drawn blocks (see QuasiMonteCarloSampling's
    # docstring) -- a coarse ballpark check, not the tight regression tolerance used for
    # MonteCarloSampling above.
    Φ_base_vals = Φ_ISHIGAMI_EXACT  # see its derivation in test/reference_values.jl
    @test all(abs.(Φ .- Φ_base_vals) .< (0.3 .* Φ_base_vals))
end

@testset "generate_permutations: reproducible given a seeded rng" begin
    p1 = generate_permutations(50, 4; rng=Xoshiro(7))
    p2 = generate_permutations(50, 4; rng=Xoshiro(7))
    @test p1 == p2
end

@testset "CallablePickAndFreezeSample(A, B; rng): reproducible given a seeded rng" begin
    # Regression test: CallablePickAndFreezeSample(A, B; ...) used to always draw its permutation
    # from the global RNG, so it was never reproducible from a seed even though every other
    # sample-drawing constructor was.
    du = Uniform(-π, π)
    A = DataFrame(rand(du, 100, 3), [:x1, :x2, :x3])
    B = DataFrame(rand(du, 100, 3), [:x1, :x2, :x3])

    S1 = CallablePickAndFreezeSample(A, B; rng=Xoshiro(9))
    S2 = CallablePickAndFreezeSample(A, B; rng=Xoshiro(9))
    @test S1.permutations == S2.permutations
    @test S1.samples == S2.samples
end

@testset "QuasiMonteCarloSampling rejects deterministic low-discrepancy algorithms" begin
    # Deterministic low-discrepancy sequences give silently wrong answers for BOTH
    # estimators, so they're rejected at construction rather than at a usage site:
    #   - pick-and-freeze: blocks sliced from one deterministic point set aren't independent
    #     -- measured cor(A₁,B₁) = 1.0000, mean|A₁-B₁| = 2.4e-4, i.e. the halves are
    #     effectively identical in coordinate 1 -- so its hybrid rows aren't valid joint
    #     draws (Ishigami: SobolSample() reported factor 1's effect as ~9e-5 against an exact
    #     6.0327 -- see Φ_ISHIGAMI_EXACT in test/reference_values.jl);
    #   - double Monte Carlo: the inner replicates are stratified, biasing the conditional
    #     sample variance (Ishigami over 8 seeds: 17% and 38% bias on two of three factors,
    #     ≈10 standard errors -- a consistent offset, not noise).
    # Randomization of the sequence does not rescue either case, hence Shift() here too.
    for alg ∈ (QMC.SobolSample(), QMC.HaltonSample(), QMC.SobolSample(QMC.Shift()),
               QMC.GoldenSample())
        @test_throws ArgumentError QuasiMonteCarloSampling(alg)
    end

    # Randomizes each dimension independently -> accepted.
    @test QuasiMonteCarloSampling(QMC.LatinHypercubeSample()) isa QuasiMonteCarloSampling
end

@testset "QuasiMonteCarloSampling: concrete algorithm field (no abstract-field boxing)" begin
    strategy = QuasiMonteCarloSampling(QMC.LatinHypercubeSample())
    @test isconcretetype(typeof(strategy))
    @test fieldtype(typeof(strategy), :algorithm) === typeof(strategy.algorithm)
end

@testset "CallableDoubleMonteCarloSample with QuasiMonteCarloSampling: correctness vs. closed form" begin
    # The QMC double-MC path had no correctness test at all -- its only assertion was
    # sum(Φ) ≈ var_y, which telescoping makes true regardless of whether the estimator
    # works. Checked here against the additive linear-Gaussian closed form, where each
    # factor's Shapley effect is exactly β_i^2 * Var(X_i).
    Random.seed!(6)
    β = [1.0, 2.0, 0.5]
    σ = [1.0, 0.5, 2.0]
    truth = (β .* σ) .^ 2

    S = CallableDoubleMonteCarloSample(
        [:x1, :x2, :x3], Normal.(0.0, σ), 2048,
        QuasiMonteCarloSampling(QMC.LatinHypercubeSample()); N_V=4096,
    )
    Φₙ, _, _ = analyze(CallableModel(x -> sum(β .* x)), S)
    Φ = SAShE.shapley_effects(Φₙ)

    @test all(abs.(Φ .- truth) .< 0.15 .* truth)
end

@testset "CallableDoubleMonteCarloSample: strategy is a required argument" begin
    factor_names = [:x1, :x2, :x3]
    dists = fill(Normal(), 3)
    @test_throws MethodError CallableDoubleMonteCarloSample(factor_names, dists, 10)
end

@testset "CallableDoubleMonteCarloSample with QuasiMonteCarloSampling: shape and step-sharing independence" begin
    Random.seed!(83)

    factor_names = [:x1, :x2, :x3, :x4]
    dists = fill(Uniform(-π, π), 4)
    strategy = QuasiMonteCarloSampling(QMC.LatinHypercubeSample())

    m = 20
    S = CallableDoubleMonteCarloSample(factor_names, dists, m, strategy; N_V=500, N_O=2, N_I=3)

    n_factors = length(factor_names)
    rows_per_step = 2 * 3
    rows_per_permutation = (n_factors - 1) * rows_per_step
    @test size(S.samples) == (500 + m * rows_per_permutation, n_factors)

    # Two different permutations sharing the same step index (j=1, i.e. the same
    # coalition *size*) must not have been handed the exact same outer/inner draws --
    # regression check for the "one combined draw across all permutations" design.
    Z = Matrix(S.samples)
    step1_perm1 = Z[(500 + 1):(500 + rows_per_step), :]
    step1_perm2 = Z[(500 + rows_per_permutation + 1):(500 + rows_per_permutation + rows_per_step), :]
    @test step1_perm1 != step1_perm2

    m2 = CallableModel(x -> sum(x))
    Φₙ, Φ²ₙ, Y = analyze(m2, S)
    Φ = SAShE.shapley_effects(Φₙ)
    @test sum(Φ) ≈ var(Y[1:500]) rtol = 0.1
end
