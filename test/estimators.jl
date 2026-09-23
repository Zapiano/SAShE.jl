@testset "estimator= kwarg: default matches explicit PickAndFreeze()" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 100, length(factor_names)
    du = Uniform(-π, π)
    ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

    X1 = DataFrame(rand(du, n_samples, n_factors), factor_names)
    X2 = DataFrame(rand(du, n_samples, n_factors), factor_names)
    model = CallableModel(ishigami, X1, X2)

    Random.seed!(7)
    Φₙ_a, Φ²ₙ_a, _ = analyze(model)
    Random.seed!(7)
    Φₙ_b, Φ²ₙ_b, _ = analyze(model; estimator=PickAndFreeze())
    @test Φₙ_a == Φₙ_b
    @test Φ²ₙ_a == Φ²ₙ_b

    S = CallableModelSample(X1, X2)
    Y = map(row -> ishigami(collect(row)), eachrow(S.samples))
    @test analyze(S, Y) == analyze(S, Y; estimator=PickAndFreeze())

    X = DataFrame(rand(du, 200, n_factors), factor_names)
    Ydata = map(row -> ishigami(collect(row)), eachrow(X))
    dmodel = DataModel(X, Ydata)
    @test analyze(dmodel, 300; rng=Xoshiro(3)) ==
        analyze(dmodel, 300; estimator=PickAndFreeze(), rng=Xoshiro(3))
end

@testset "estimator= kwarg: PickAndFreeze is exported and constructible" begin
    @test PickAndFreeze() isa SAShE.EstimationMethod
end
