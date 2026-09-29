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
    S = CallablePickAndFreezeSample(X1, X2)
    Φₙ, Φ²ₙ = analyze(m, S)
    @test size(Φₙ) == (n_factors, n_samples)
end

@testset "PickAndFreeze is exported and constructible" begin
    @test PickAndFreeze() isa SAShE.EstimationMethod
end

@testset "analyze(s, Y): matches analyze(m, s) exactly, given the same Y" begin
    # Tiny sizes on purpose -- this only needs to confirm the two code paths agree bit for
    # bit, not re-check statistical correctness (already covered by the Ishigami tests).
    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 20, length(factor_names)
    du = Uniform(-π, π)
    ishigami(x) = (1 + 0.1x[3]^4) * sin(x[1]) + 7 * sin(x[2])^2
    m = CallableModel(ishigami)

    X1 = DataFrame(rand(du, n_samples, n_factors), factor_names)
    X2 = DataFrame(rand(du, n_samples, n_factors), factor_names)
    S_pf = CallablePickAndFreezeSample(X1, X2)
    Φₙ, Φ²ₙ, Yₙ = analyze(m, S_pf)
    Φₙ2, Φ²ₙ2, Yₙ2 = analyze(S_pf, Yₙ)
    @test Φₙ == Φₙ2
    @test Φ²ₙ == Φ²ₙ2
    @test Yₙ == Yₙ2
    @test_throws ArgumentError analyze(S_pf, Yₙ[1:(end - 1)])

    S_dmc = CallableDoubleMonteCarloSample(
        factor_names, fill(du, n_factors), 20, MonteCarloSampling(); N_V=50
    )
    Φₙ, Φ²ₙ, Yₙ = analyze(m, S_dmc)
    Φₙ2, Φ²ₙ2, Yₙ2 = analyze(S_dmc, Yₙ)
    @test Φₙ == Φₙ2
    @test Φ²ₙ == Φ²ₙ2
    @test Yₙ == Yₙ2
    @test_throws ArgumentError analyze(S_dmc, Yₙ[1:(end - 1)])
end
