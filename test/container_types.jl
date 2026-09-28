@testset "DataModel: analyze is deterministic given a seed" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 200, length(factor_names)
    du = Uniform(-π, π)
    ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

    X = DataFrame(rand(du, n_samples, n_factors), factor_names)
    Y = map(row -> ishigami(collect(row)), eachrow(X))
    model = DataModel(X, Y)

    Φₙ_a, Φ²ₙ_a = SAShE.analyze(model, 500, PickAndFreeze(); rng=Xoshiro(1))
    Φₙ_b, Φ²ₙ_b = SAShE.analyze(model, 500, PickAndFreeze(); rng=Xoshiro(1))

    @test Φₙ_a == Φₙ_b
    @test Φ²ₙ_a == Φ²ₙ_b
end

@testset "DataModel: rejects mismatched X/Y row counts" begin
    factor_names = [:x1, :x2, :x3]
    du = Uniform(-π, π)

    X = DataFrame(rand(du, 200, length(factor_names)), factor_names)
    Y = rand(du, 199)

    @test_throws ArgumentError DataModel(X, Y)
end

@testset "MixModel: constructs" begin
    # MixModel wraps only func, like CallableModel -- no X/Y to validate here; see
    # test/mix_model.jl for MixPickAndFreezeSample/MixDoubleMonteCarloSample's coverage of
    # that (they hold the dataset), and for analyze's dispatch-by-sample-type behaviour.
    ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2
    mm = MixModel(ishigami)
    @test mm.func === ishigami
    @test_throws MethodError SAShE.analyze(mm)
end
