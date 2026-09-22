using LinearAlgebra: cholesky, \

"""
Exact Shapley effects for a linear Gaussian model `Y = β'X`, `X ~ N(0, Γ)`, computed by
brute-force enumeration of the Owen/Shapley decomposition ([1], Eq. 4), used as ground
truth for the correctness check below. Not part of the package API — a test-only reference
implementation, independent of the code under test.
"""
function _linear_gaussian_theoretical_shapley(β::Vector{Float64}, Γ::Matrix{Float64})
    p = length(β)
    all_idx = collect(1:p)
    var_y = sum(β .* (Γ * β))

    # E(X_{-u} | X_u) = Γ_{-u,u} Γ_{u,u}^{-1} X_u  (zero-mean Gaussian), so
    # E(Y | X_u) = c_u' X_u  with  c_u = β_u + Γ_{u,u}^{-1} Γ_{u,-u} β_{-u}.
    function V(u::Vector{Int64})
        isempty(u) && return 0.0
        length(u) == p && return var_y
        nu = setdiff(all_idx, u)
        Γuu = Γ[u, u]
        Γu_nu = Γ[u, nu]
        c = β[u] .+ (Γuu \ (Γu_nu * β[nu]))
        return sum(c .* (Γuu * c))
    end

    subsets(v::Vector{Int64}) = [
        v[[Bool((mask >> (j - 1)) & 1) for j ∈ eachindex(v)]] for mask ∈ 0:(2^length(v) - 1)
    ]

    η = zeros(p)
    for i ∈ 1:p
        others = setdiff(all_idx, [i])
        for u ∈ subsets(others)
            w = 1 / binomial(p - 1, length(u))
            η[i] += w * (V(vcat(u, i)) - V(u))
        end
        η[i] /= p
    end
    return η
end

@testset "Nearest-neighbour Shapley effects: axiom check — efficiency" begin
    Random.seed!(11)

    n_samples, n_factors = 500, 4
    X = DataFrame(randn(n_samples, n_factors), [:x1, :x2, :x3, :x4])
    Y = collect(sum.(eachrow(X)) .+ 0.3 .* X.x1 .* X.x2)  # a bit of nonlinearity too

    Φₙ, Φ²ₙ = SAShE.analyze(X, Y, 500)
    Φ = SAShE.shapley_effects(Φₙ)

    # Holds exactly (not just approximately) by construction: the last position of every
    # permutation walk uses the known var(Y) directly, never a nearest-neighbour estimate.
    @test sum(Φ) ≈ var(Y)
end

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

    Φₙ, Φ²ₙ = SAShE.analyze(X, Y, 6000)
    Φ = SAShE.shapley_effects(Φₙ)

    @test all(abs.(Φ .- η_theoretical) .< 0.15 .* η_theoretical)
end

@testset "Nearest-neighbour Shapley effects: regression vs. exact estimator (independent Ishigami)" begin
    Random.seed!(33)

    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 8000, 3
    du = Uniform(-π, π)

    X = DataFrame(rand(du, n_samples, n_factors), factor_names)
    Y = map(x -> ishigami(collect(x)), eachrow(X))

    Φₙ, Φ²ₙ = SAShE.analyze(X, Y, 6000)
    Φ = SAShE.shapley_effects(Φₙ)

    # Same theoretical baseline already validated (against the exact estimator) in
    # test/ishigami.jl — the nearest-neighbour estimator is expected to be noisier (Pick-and-Freeze has
    # higher variance than double-MC, per [1]'s own findings), hence the looser tolerance
    # than the exact-estimator test uses. See docs/src/references.md for citation [1].
    Φ_base_vals = [6.17, 6.08, 1.64]
    @test all(abs.(Φ .- Φ_base_vals) .< 0.3 .* Φ_base_vals)
end

@testset "Nearest-neighbour Shapley effects: error path for N_I > N" begin
    X = DataFrame(randn(2, 3), [:x1, :x2, :x3])  # only 2 rows: N_I=2 needs ≥3
    Y = collect(sum.(eachrow(X)))

    @test_throws ArgumentError SAShE.analyze(X, Y, 10)
end
