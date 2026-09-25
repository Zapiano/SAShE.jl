@testset "DataModel: analyze requires an explicit estimator, no default" begin
    factor_names = [:x1, :x2, :x3]
    n_factors = length(factor_names)
    du = Uniform(-π, π)
    ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

    X = DataFrame(rand(du, 200, n_factors), factor_names)
    Ydata = map(row -> ishigami(collect(row)), eachrow(X))
    dmodel = DataModel(X, Ydata)

    @test_throws MethodError analyze(dmodel, 300; rng=Xoshiro(3))

    Φₙ_a, Φ²ₙ_a = analyze(dmodel, 300, PickAndFreeze(); rng=Xoshiro(3))
    Φₙ_b, Φ²ₙ_b = analyze(dmodel, 300, PickAndFreeze(); rng=Xoshiro(3))
    @test Φₙ_a == Φₙ_b
end

@testset "analyze(m::CallableModel, s): estimator is implied by the sample's type" begin
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 100, length(factor_names)
    du = Uniform(-π, π)
    ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2

    X1 = DataFrame(rand(du, n_samples, n_factors), factor_names)
    X2 = DataFrame(rand(du, n_samples, n_factors), factor_names)
    m = CallableModel(ishigami)

    # No `estimator=` keyword exists here at all -- calling analyze(m, s) is unambiguous.
    S = PickAndFreezeSample(X1, X2)
    Φₙ, Φ²ₙ = analyze(m, S)
    @test size(Φₙ) == (n_factors, n_samples)
end

@testset "PickAndFreeze is exported and constructible" begin
    @test PickAndFreeze() isa SAShE.EstimationMethod
end
