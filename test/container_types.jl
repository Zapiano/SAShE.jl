@testset "DataModel: analyze is deterministic given a seed" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 200, length(factor_names)
    du = Uniform(-π, π)
    ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

    X = DataFrame(rand(du, n_samples, n_factors), factor_names)
    Y = map(row -> ishigami(collect(row)), eachrow(X))
    model = DataModel(X, Y)

    Φₙ_a, Φ²ₙ_a = SAShE.analyze(model, 500; rng=Xoshiro(1))
    Φₙ_b, Φ²ₙ_b = SAShE.analyze(model, 500; rng=Xoshiro(1))

    @test Φₙ_a == Φₙ_b
    @test Φ²ₙ_a == Φ²ₙ_b
end

@testset "MixModel: constructs, has no estimator yet" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 20, length(factor_names)
    du = Uniform(-π, π)
    ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

    X = DataFrame(rand(du, n_samples, n_factors), factor_names)
    Y = map(row -> ishigami(collect(row)), eachrow(X))

    mm = MixModel(ishigami, X, Y)
    @test mm.func === ishigami
    @test_throws MethodError SAShE.analyze(mm)
end
