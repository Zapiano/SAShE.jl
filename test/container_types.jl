@testset "DataModel: equivalent to the loose (X, Y) call" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 200, length(factor_names)
    du = Uniform(-π, π)
    ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

    X = DataFrame(rand(du, n_samples, n_factors), factor_names)
    Y = map(row -> ishigami(collect(row)), eachrow(X))

    Φₙ_direct, Φ²ₙ_direct = SAShE.analyze(X, Y, 500; rng=Xoshiro(1))
    Φₙ_model, Φ²ₙ_model = SAShE.analyze(DataModel(X, Y), 500; rng=Xoshiro(1))

    @test Φₙ_direct == Φₙ_model
    @test Φ²ₙ_direct == Φ²ₙ_model
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
