"""
Ground-truth values shared across the test suite. Included first by `runtests.jl`, and
required by any partial run (the suite is conventionally run with `ishigami.jl` separated out
from the rest — see issue #14 — so shared constants must not live in either half).
"""

using LinearAlgebra: cholesky, \

"""
    ishigami(X; a=7.0, b=0.1)

The Ishigami test function, `(1 + b·x₃⁴)·sin(x₁) + a·sin²(x₂)`.
"""
function ishigami(X::Vector{Float64}; a::Float64=7.0, b::Float64=0.1)
    return (1 + b * X[3]^4) * sin(X[1]) + a * (sin(X[2]))^2
end

"""
Exact Shapley effects of [`ishigami`](@ref) with `a=7, b=0.1`, `Xᵢ ~ U(-π, π)` iid.

Derived, not fitted. The ANOVA decomposition has only three non-zero components:

    σ²₁  = ½(1 + bπ⁴/5)²      = 4.345886
    σ²₂  = a²/8               = 6.125
    σ²₁₃ = b²π⁸(1/18 − 1/50)  = 3.373701
    σ²₃ = σ²₁₂ = σ²₂₃ = σ²₁₂₃ = 0

Shapley effects then follow from Owen's decomposition, `φᵢ = Σ_{u ∋ i} σ²_u / |u|`:

    φ₁ = σ²₁ + σ²₁₃/2,   φ₂ = σ²₂,   φ₃ = σ²₁₃/2

Cross-checked two independent ways: (1) brute-force numerical integration of `Var(E[Y|X_u])`
for all 8 subsets on a 400³ midpoint grid, followed by the exact Shapley average over all 6
permutations — `[6.03246, 6.12500, 1.68669]` with `Var[Y] = 13.844152`; (2) a 4·10⁶-sample
plain Monte Carlo estimate of `Var[Y]` — `13.8414`. Both agree with the closed form to ~4
significant figures; the residual is grid/MC discretization, not disagreement.

NOTE: the suite previously used `[6.17, 6.08, 1.64]` here and labelled it "theoretical". It
is not — it is off by −2.3%, +0.7%, −2.8%, and its sum (13.89) overshoots `Var[Y]`. Also
worth knowing when reading tolerances: `φ₂` is the *largest* effect, not `φ₁`.
"""
const Φ_ISHIGAMI_EXACT = [6.032738, 6.125, 1.686851]

"Exact `Var[Y]` for [`ishigami`](@ref) with `a=7, b=0.1`, `Xᵢ ~ U(-π, π)` iid."
const VAR_Y_ISHIGAMI_EXACT = 13.844588

"""
    _linear_gaussian_theoretical_shapley(β, Γ)::Vector{Float64}

Exact Shapley effects for a linear Gaussian model `Y = β'X`, `X ~ N(0, Γ)`, computed by
brute-force enumeration of the Owen/Shapley decomposition ([1], Eq. 4), used as ground truth
for correctness checks against dependent-factor estimators (nearest-neighbour and `MixModel`
both approximate conditional structure via the dataset itself, so this is the reference for
either). Not part of the package API — a test-only reference implementation, independent of
the code under test.
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
