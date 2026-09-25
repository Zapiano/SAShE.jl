@testset "MixModel + MixPickAndFreezeSample: correctness vs. independent-linear closed form" begin
    # Averaged over several independent datasets rather than checking a single seeded run --
    # same reasoning as the Ishigami regression test below (and
    # test/nearest_neighbour_shapley_effects.jl's Ishigami test, which this mirrors): measured
    # single-seed max relative error peaked at 0.139 across 8 seeds (right at the 0.15
    # tolerance's edge, 0/8 failed outright but with little margin), so a single-run check here
    # is closer to a coin flip than the tolerance suggests. 5-replicate averaging is the same
    # fix applied there.
    β = [1.0, 2.0, 0.5]
    σ = [1.0, 0.5, 2.0]
    dists = Normal.(0.0, σ)
    model(x) = sum(β .* x)
    η_theoretical = (β .* σ) .^ 2
    n_samples, n_replicates, n_permutations = 8000, 5, 4000

    mm = MixModel(model)
    Φ_total = zeros(3)
    for seed ∈ 1:n_replicates
        Random.seed!(30 + seed)
        X = DataFrame(hcat(rand.(dists, n_samples)...), [:x1, :x2, :x3])
        Y = map(row -> model(collect(row)), eachrow(X))
        Φₙ, _, _ = analyze(mm, MixPickAndFreezeSample(X, Y, n_permutations))
        Φ_total .+= SAShE.shapley_effects(Φₙ)
    end
    Φ = Φ_total ./ n_replicates

    @test all(abs.(Φ .- η_theoretical) .< 0.15 .* η_theoretical)
end

@testset "MixModel + MixDoubleMonteCarloSample: correctness vs. independent-linear closed form" begin
    # Measured single-seed max relative error peaked at 0.077 across 8 seeds -- comfortable
    # margin under 0.15, but averaged anyway for consistency with the PF variant above.
    β = [1.0, 2.0, 0.5]
    σ = [1.0, 0.5, 2.0]
    dists = Normal.(0.0, σ)
    model(x) = sum(β .* x)
    η_theoretical = (β .* σ) .^ 2
    n_samples, n_replicates, n_permutations = 8000, 5, 4000

    mm = MixModel(model)
    Φ_total = zeros(3)
    for seed ∈ 1:n_replicates
        Random.seed!(40 + seed)
        X = DataFrame(hcat(rand.(dists, n_samples)...), [:x1, :x2, :x3])
        Y = map(row -> model(collect(row)), eachrow(X))
        Φₙ, _, _ = analyze(mm, MixDoubleMonteCarloSample(X, Y, n_permutations))
        Φ_total .+= SAShE.shapley_effects(Φₙ)
    end
    Φ = Φ_total ./ n_replicates

    @test all(abs.(Φ .- η_theoretical) .< 0.15 .* η_theoretical)
end

@testset "MixModel + MixPickAndFreezeSample: correctness vs. dependent-linear-Gaussian closed form" begin
    # Measured single-seed max relative error hit 0.153 in 1 of 8 seeds -- this one actually
    # failed empirically, not just theoretically close to the edge. 5-replicate averaging
    # brings the worst case comfortably under tolerance.
    p = 3
    β = [1.0, 1.0, 1.0]
    # x1, x3 correlated (ρ=0.6); x2 independent — same benchmark shape used for DataModel in
    # test/nearest_neighbour_shapley_effects.jl. See docs/src/references.md for citation [1].
    Γ = [1.0 0.0 0.6
         0.0 1.0 0.0
         0.6 0.0 1.0]
    η_theoretical = _linear_gaussian_theoretical_shapley(β, Γ)
    n_samples, n_replicates, n_permutations = 8000, 5, 6000

    mm = MixModel(x -> sum(β .* x))
    Φ_total = zeros(3)
    for seed ∈ 1:n_replicates
        Random.seed!(50 + seed)
        Xm = randn(n_samples, p) * cholesky(Γ).U
        X = DataFrame(Xm, [:x1, :x2, :x3])
        Y = collect(Xm * β)
        Φₙ, _, _ = analyze(mm, MixPickAndFreezeSample(X, Y, n_permutations))
        Φ_total .+= SAShE.shapley_effects(Φₙ)
    end
    Φ = Φ_total ./ n_replicates

    @test all(abs.(Φ .- η_theoretical) .< 0.15 .* η_theoretical)
end

@testset "MixModel + MixDoubleMonteCarloSample: correctness vs. dependent-linear-Gaussian closed form" begin
    # Measured single-seed max relative error peaked at 0.061 across 8 seeds -- comfortable
    # margin, averaged anyway for consistency with the PF variant above.
    p = 3
    β = [1.0, 1.0, 1.0]
    Γ = [1.0 0.0 0.6
         0.0 1.0 0.0
         0.6 0.0 1.0]
    η_theoretical = _linear_gaussian_theoretical_shapley(β, Γ)
    n_samples, n_replicates, n_permutations = 8000, 5, 6000

    mm = MixModel(x -> sum(β .* x))
    Φ_total = zeros(3)
    for seed ∈ 1:n_replicates
        Random.seed!(60 + seed)
        Xm = randn(n_samples, p) * cholesky(Γ).U
        X = DataFrame(Xm, [:x1, :x2, :x3])
        Y = collect(Xm * β)
        Φₙ, _, _ = analyze(mm, MixDoubleMonteCarloSample(X, Y, n_permutations))
        Φ_total .+= SAShE.shapley_effects(Φₙ)
    end
    Φ = Φ_total ./ n_replicates

    @test all(abs.(Φ .- η_theoretical) .< 0.15 .* η_theoretical)
end

@testset "MixModel: regression vs. exact estimator (independent Ishigami)" begin
    # Averaged over several independent datasets, not a single seeded run -- see the
    # equivalent DataModel test in test/nearest_neighbour_shapley_effects.jl for why: this
    # family of estimators has per-run spread comparable to the smallest effect being
    # measured, so a single-run check against a loose tolerance still has a real chance of
    # failing by chance rather than by a genuine problem.
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 4000, length(factor_names)
    n_replicates, n_permutations = 5, 2000
    du = Uniform(-π, π)

    mm = MixModel(ishigami)
    Φ_pf_total = zeros(n_factors)
    Φ_dmc_total = zeros(n_factors)
    for seed ∈ 1:n_replicates
        Random.seed!(seed)
        X = DataFrame(rand(du, n_samples, n_factors), factor_names)
        Y = map(x -> ishigami(collect(x)), eachrow(X))

        Φₙ_pf, _ = analyze(mm, MixPickAndFreezeSample(X, Y, n_permutations))
        Φ_pf_total .+= SAShE.shapley_effects(Φₙ_pf)

        Φₙ_dmc, _ = analyze(mm, MixDoubleMonteCarloSample(X, Y, n_permutations))
        Φ_dmc_total .+= SAShE.shapley_effects(Φₙ_dmc)
    end

    Φ_pf = Φ_pf_total ./ n_replicates
    Φ_dmc = Φ_dmc_total ./ n_replicates

    @test all(abs.(Φ_pf .- Φ_ISHIGAMI_EXACT) .< 0.3 .* Φ_ISHIGAMI_EXACT)
    @test all(abs.(Φ_dmc .- Φ_ISHIGAMI_EXACT) .< 0.3 .* Φ_ISHIGAMI_EXACT)
end

@testset "MixModel: analyze is deterministic given a seeded sample and rng" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 200, length(factor_names)
    du = Uniform(-π, π)

    X = DataFrame(rand(du, n_samples, n_factors), factor_names)
    Y = map(row -> ishigami(collect(row)), eachrow(X))
    mm = MixModel(ishigami)

    S_pf = MixPickAndFreezeSample(X, Y, 300; rng=Xoshiro(1))
    Φₙ_pf_a, _ = analyze(mm, S_pf; rng=Xoshiro(11))
    Φₙ_pf_b, _ = analyze(mm, S_pf; rng=Xoshiro(11))
    @test Φₙ_pf_a == Φₙ_pf_b

    S_dmc = MixDoubleMonteCarloSample(X, Y, 300; rng=Xoshiro(2))
    Φₙ_dmc_a, _ = analyze(mm, S_dmc; rng=Xoshiro(22))
    Φₙ_dmc_b, _ = analyze(mm, S_dmc; rng=Xoshiro(22))
    @test Φₙ_dmc_a == Φₙ_dmc_b

    # The sample itself is reproducible from a seed, same as CallablePickAndFreezeSample/
    # CallableDoubleMonteCarloSample -- not just the analysis of a fixed sample.
    S_pf2 = MixPickAndFreezeSample(X, Y, 300; rng=Xoshiro(1))
    @test S_pf.permutations == S_pf2.permutations
    @test S_pf.references == S_pf2.references
end

@testset "MixDoubleMonteCarloSample: N_I must be ≥ 2" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 200, length(factor_names)
    du = Uniform(-π, π)

    X = DataFrame(rand(du, n_samples, n_factors), factor_names)
    Y = map(row -> ishigami(collect(row)), eachrow(X))

    @test_throws ArgumentError MixDoubleMonteCarloSample(X, Y, 10; N_I=1)
end

@testset "MixPickAndFreezeSample/MixDoubleMonteCarloSample: reject malformed direct construction" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 200, length(factor_names)
    du = Uniform(-π, π)

    X = DataFrame(rand(du, n_samples, n_factors), factor_names)
    Y = map(row -> ishigami(collect(row)), eachrow(X))

    perms = generate_permutations(50, n_factors)
    refs = rand(1:n_samples, 50)

    @test MixPickAndFreezeSample(X, Y, perms, refs) isa MixPickAndFreezeSample

    # X/Y row-count mismatch.
    @test_throws ArgumentError MixPickAndFreezeSample(X, Y[1:(end - 1)], perms, refs)
    # permutations has the wrong number of columns.
    @test_throws ArgumentError MixPickAndFreezeSample(X, Y, generate_permutations(50, 2), refs)
    # references has the wrong length.
    @test_throws ArgumentError MixPickAndFreezeSample(X, Y, perms, refs[1:(end - 1)])
    # references indexes outside 1:n_samples.
    bad_refs = copy(refs)
    bad_refs[1] = n_samples + 1
    @test_throws ArgumentError MixPickAndFreezeSample(X, Y, perms, bad_refs)

    # A row that's the right shape but not a genuine permutation of 1:n_factors.
    bad_perms = copy(perms)
    bad_perms[1, :] .= 1
    @test_throws ArgumentError MixPickAndFreezeSample(X, Y, bad_perms, refs)
    @test_throws ArgumentError MixDoubleMonteCarloSample(X, Y, bad_perms, refs, 3)

    @test MixDoubleMonteCarloSample(X, Y, perms, refs, 3) isa MixDoubleMonteCarloSample
    @test_throws ArgumentError MixDoubleMonteCarloSample(X, Y[1:(end - 1)], perms, refs, 3)
end

@testset "MixModel: analyze rejects a dataset where Y doesn't match func" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 200, length(factor_names)
    du = Uniform(-π, π)

    X = DataFrame(rand(du, n_samples, n_factors), factor_names)
    Y = map(row -> ishigami(collect(row)), eachrow(X))
    Y_wrong = Y .+ 1.0  # every row now disagrees with `ishigami`

    mm = MixModel(ishigami)

    S_pf = MixPickAndFreezeSample(X, Y_wrong, 50; rng=Xoshiro(1))
    @test_throws ArgumentError analyze(mm, S_pf; rng=Xoshiro(1))

    S_dmc = MixDoubleMonteCarloSample(X, Y_wrong, 50; rng=Xoshiro(1))
    @test_throws ArgumentError analyze(mm, S_dmc; rng=Xoshiro(1))

    # A dataset that does match func passes the check and analyze proceeds normally.
    S_pf_ok = MixPickAndFreezeSample(X, Y, 50; rng=Xoshiro(1))
    @test analyze(mm, S_pf_ok; rng=Xoshiro(1)) isa Tuple
end

@testset "MixModel: analyze's n_checks controls the consistency check's cost" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 200, length(factor_names)
    du = Uniform(-π, π)

    X = DataFrame(rand(du, n_samples, n_factors), factor_names)
    Y = map(row -> ishigami(collect(row)), eachrow(X))
    Y_wrong = zeros(n_samples)  # disagrees with `ishigami` at every row

    mm = MixModel(ishigami)
    S_pf_wrong = MixPickAndFreezeSample(X, Y_wrong, 50; rng=Xoshiro(1))
    S_dmc_wrong = MixDoubleMonteCarloSample(X, Y_wrong, 50; rng=Xoshiro(1))

    # n_checks=0 skips the check entirely -- a mismatched dataset no longer errors here (it
    # will still bias the terminal W silently, which is the accepted cost of opting out).
    @test analyze(mm, S_pf_wrong; rng=Xoshiro(1), n_checks=0) isa Tuple
    @test analyze(mm, S_dmc_wrong; rng=Xoshiro(1), n_checks=0) isa Tuple

    # On a dataset that does match func, n_checks controls exactly how many `func` calls the
    # check itself makes (a mismatched dataset instead throws on the very first check, so
    # counting calls there would only ever show 1, regardless of n_checks). Compare the same
    # sample (so the walk's own call count is identical both times) at two `n_checks` values;
    # the difference in total calls must equal the difference in `n_checks`.
    calls = Ref(0)
    counted(x) = (calls[] += 1; ishigami(x))
    mm_counted = MixModel(counted)
    S_pf = MixPickAndFreezeSample(X, Y, 5; rng=Xoshiro(1))

    calls[] = 0
    analyze(mm_counted, S_pf; rng=Xoshiro(1), n_checks=5)
    calls_at_5 = calls[]

    calls[] = 0
    analyze(mm_counted, S_pf; rng=Xoshiro(1), n_checks=1)
    calls_at_1 = calls[]

    @test calls_at_5 - calls_at_1 == 4
end

@testset "_nearest_neighbour_mix_pick_freeze/_nearest_neighbour_mix_double_monte_carlo: exact values on a hand-built, tie-free dataset" begin
    # Rows chosen so that, restricted to either `u = [1]` (column 1 alone) or `notU = [2, 3]`
    # (columns 2-3), the nearest-neighbour ordering relative to row 1 is unambiguous (no
    # exact ties) -- this pins the coordinate-subset choice and the `N_I -> k` off-by-one
    # translation that no end-to-end correctness test (linear or Ishigami closed form) can
    # detect, since both are statistically invisible in aggregate (a wrong-but-consistent
    # neighbour count/subset still converges to the right answer, just with different
    # variance -- see the review this test responds to).
    X = DataFrame(
        [0.0 0.0 0.0
         1.0 5.0 9.0
         2.0 6.0 1.0
         3.0 1.0 8.0
         4.0 9.0 2.0],
        [:x1, :x2, :x3],
    )
    Xm = Matrix(X)
    func(x) = x[1] + 2 * x[2] + 3 * x[3]  # linear -- trivial to hand-evaluate
    s = 1
    Ȳ = 1.0  # arbitrary; only tests that it's subtracted as Ȳ², not recomputed internally

    @testset "_nearest_neighbour_mix_pick_freeze" begin
        # u = [1]: nearest neighbour by column 1 alone (values 0,1,2,3,4 for rows 1:5) is
        # row 2 (distance 1). hybrid = x_s with columns [2, 3] taken from row 2.
        W, calls = SAShE._nearest_neighbour_mix_pick_freeze(func, Xm, Ȳ, s, [1])
        f_x_s = func([0.0, 0.0, 0.0])       # = 0.0
        f_hybrid = func([0.0, 5.0, 9.0])    # = 0 + 10 + 27 = 37.0
        @test calls == [f_x_s, f_hybrid]
        @test W ≈ f_x_s * f_hybrid - Ȳ^2
        @test W ≈ 0.0 * 37.0 - 1.0
    end

    @testset "_nearest_neighbour_mix_double_monte_carlo" begin
        # u = [1], N_I = 3: nearest 2 external neighbours by columns [2, 3] alone (standardized
        # Euclidean) are row 3 then row 4 (hand-verified against row 2 and row 5). Each hybrid
        # takes column 1 from the neighbour, columns [2, 3] from x_s.
        W, calls = SAShE._nearest_neighbour_mix_double_monte_carlo(func, Xm, s, [1], 3)
        f_x_s = func([0.0, 0.0, 0.0])        # = 0.0
        f_nb3 = func([2.0, 0.0, 0.0])        # x1 from row 3 = 2.0
        f_nb4 = func([3.0, 0.0, 0.0])        # x1 from row 4 = 3.0
        @test calls == [f_x_s, f_nb3, f_nb4]
        @test W ≈ var([f_x_s, f_nb3, f_nb4])  # Bessel-corrected, matches Broto Eq. (17)
        @test W ≈ 7 / 3
    end
end

@testset "MixModel: analyze's Yₙ has the documented length and includes the reference-row calls" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 200, length(factor_names)
    n_permutations = 25
    du = Uniform(-π, π)

    X = DataFrame(rand(du, n_samples, n_factors), factor_names)
    Y = map(row -> ishigami(collect(row)), eachrow(X))
    mm = MixModel(ishigami)

    S_pf = MixPickAndFreezeSample(X, Y, n_permutations; rng=Xoshiro(1))
    Φₙ_pf, _, Yₙ_pf = analyze(mm, S_pf; rng=Xoshiro(1))
    @test length(Yₙ_pf) == n_permutations * (n_factors - 1) * 2

    N_I = 3
    S_dmc = MixDoubleMonteCarloSample(X, Y, n_permutations; N_I=N_I, rng=Xoshiro(2))
    Φₙ_dmc, _, Yₙ_dmc = analyze(mm, S_dmc; rng=Xoshiro(2))
    @test length(Yₙ_dmc) == n_permutations * (n_factors - 1) * N_I

    # Every permutation's first coalition step evaluates `func` at the reference row itself
    # (verbatim, per _nearest_neighbour_mix_pick_freeze/_nearest_neighbour_mix_double_monte_carlo's
    # docstrings) as the first of its calls -- so those values must appear in Yₙ.
    for m ∈ 1:n_permutations
        s = S_pf.references[m]
        @test ishigami(collect(X[s, :])) ∈ Yₙ_pf
    end
end
