using Random: AbstractRNG, default_rng, randperm
using Statistics: mean, var

"""
    DataModel(X::DataFrame, Y::Vector)

Container for the "fixed dataset, no callable model" case: wraps the `(X, Y)` pair
analyzed by [`analyze(model::DataModel, n_permutations::Integer)`](@ref).

# Arguments
- `X` : Fixed sample of inputs, one row per observation, one column per factor.
- `Y` : Corresponding outputs, `Y[n]` is the already-known output for row `n` of `X`.
"""
struct DataModel
    X::DataFrame
    Y::Vector
end

"""
    analyze(model::DataModel, n_permutations::Integer; rng=default_rng())::Tuple{Matrix{Float64},Matrix{Float64}}

Estimate Shapley effects from a fixed dataset `(X, Y)` alone — no callable model, no
joint-distribution model — using [1]'s nearest-neighbour "knn" Pick-and-Freeze estimator
(§6.1.2, Eq. 23) combined with [2]'s random-permutation W-aggregation procedure (Eq. 14).
Zero new function evaluations: every conditional-element estimate reuses `Y` values already
present in the dataset. See the [Dataset-only workflow](@ref) page for a full walk-through
and the estimator's caveats.

For each of `n_permutations` random permutations of the factors, a single random reference
row is drawn from `X`; walking the permutation builds nested coalitions
`u_1 ⊂ u_2 ⊂ ... ⊂ u_p = [1:p]`, and each intermediate `V_{u_i}` is estimated via
[`_nearest_neighbour_pick_freeze`](@ref). Consecutive differences accumulate into each factor's
Shapley-effect contribution.

# Arguments
- `model` : The dataset to analyze, wrapped in a [`DataModel`](@ref).
- `n_permutations` : Number of random permutations to average over — an accuracy/budget
  knob; more permutations means lower estimator variance.
- `rng` : Random number generator (keyword, optional).

# Returns
Tuple `(Φₙ, Φ²ₙ)`, in the same shape as [`analyze(S::CallableModelSample, Y::Vector)`](@ref)
returns — pass to [`shapley_effects`](@ref) or [`confint`](@ref) as usual.

See the [References](@ref) page for the full citations behind [1] and [2].
"""
function analyze(
    model::DataModel, n_permutations::Integer; rng::AbstractRNG=default_rng()
)::Tuple{Matrix{Float64}, Matrix{Float64}}
    Xm = Matrix(model.X)
    Y = model.Y
    n_samples, n_factors = size(Xm)
    Ȳ = mean(Y)
    var_y = var(Y)

    Φₙ_increments = zeros(n_factors, n_permutations)
    Φₙ²_increments = zeros(n_factors, n_permutations)

    for m ∈ 1:n_permutations
        σ = randperm(rng, n_factors)
        s = rand(rng, 1:n_samples)

        prevW = 0.0
        u = Int64[]
        for i ∈ 1:n_factors
            push!(u, σ[i])
            W = if i == n_factors
                var_y
            else
                _nearest_neighbour_pick_freeze(Xm, Y, Ȳ, s, u; rng=rng)
            end

            Δ = W - prevW
            Φₙ_increments[σ[i], m] = Δ / n_permutations
            Φₙ²_increments[σ[i], m] = Δ^2 / n_permutations

            prevW = W
        end
    end

    return Φₙ_increments, Φₙ²_increments
end

"""
    _nearest_neighbour_pick_freeze(X, Y, Ȳ, s, u; rng=default_rng())::Float64

The V̂^knn_{u,s,PF} estimator of [1], §6.1.2, Eq. (23): the product of the outputs of the
two nearest neighbours (restricted to coordinates `u`) of row `s` of `X`, minus the squared
sample mean of `Y`. Uses only values already present in `(X, Y)` — no new function
evaluations.

The neighbour count is fixed at 2, not a tunable accuracy/cost knob — see the
[Dataset-only workflow](@ref) page's Caveats section for why.

# Arguments
- `X` : Sample matrix, rows = observations, columns = coordinates.
- `Y` : Corresponding outputs, `Y[n]` is the already-known output for row `n` of `X`.
- `Ȳ` : Sample mean of `Y` (passed in so callers don't recompute it on every call).
- `s` : Row index, in `1:size(X, 1)`, of the query point.
- `u` : Coalition (column indices), nonempty and a proper subset of the factors, to
  restrict the nearest-neighbour search to.
- `rng` : Random number generator used for tie-breaking (keyword, optional).

# Returns
A single `Float64`: the estimate of V_u = Var(E(Y | X_u)).

See the [References](@ref) page for the full citation behind [1].
"""
function _nearest_neighbour_pick_freeze(
    X::AbstractMatrix, Y::AbstractVector{<:Real}, Ȳ::Real, s::Integer,
    u::AbstractVector{<:Integer}; rng::AbstractRNG=default_rng(),
)::Float64
    # Fixed by the estimator's own structure, not a tunable accuracy knob — see docstring.
    N_I = 2
    idx = _nearest_neighbour_indices(X, s, u, N_I; rng=rng)
    return Y[idx[1]] * Y[idx[2]] - Ȳ^2
end
