using NearestNeighbors: BallTree, knn
using Random: AbstractRNG, default_rng, randperm
using Statistics: mean, std, var

"""
    _standardize_columns(X::AbstractMatrix)::Matrix{Float64}

Z-score each column (coordinate) of `X` by its sample standard deviation.

Coordinates handed to nearest-neighbour search are not assumed to share a common scale or
to be independent of each other; unweighted Euclidean distance would otherwise let
whichever coordinate has the largest variance dominate the search. Standardizing accounts
for scale but not correlation between coordinates — a Mahalanobis-distance option
(accounting for both) is deferred to a later addition.

# Arguments
- `X` : Sample matrix, rows = observations, columns = coordinates.

# Returns
Matrix of the same size as `X`, with each column z-scored by its own sample standard
deviation.
"""
function _standardize_columns(X::AbstractMatrix)::Matrix{Float64}
    σ = std(X; dims=1)
    return X ./ σ
end

"""
    _nearest_neighbour_indices(X, l, coords, k; rng=default_rng())::Vector{Int64}

Indices, in `1:size(X, 1)`, of the `k` nearest neighbours of row `l` of `X`, restricted to
the columns in `coords`, excluding `l` itself. This is the k^v_N(l, n) primitive of
[1], §6: the n-th nearest neighbour of sample point `l`, restricted to coordinate subset
`v`, among the sample restricted to the same coordinates.

Distance is standardized Euclidean (see [`_standardize_columns`](@ref)). Ties at the k-th
nearest neighbour's distance are broken uniformly at random ([1]'s Assumption 7), rather
than by whichever order the search happens to return — this matters for discrete numeric
coordinates, where exact ties are possible even without categorical data. Tie-breaking
considers every point queried from the tree; to keep this exact rather than approximate, a
small buffer beyond `k` is queried so that ties just past the k-th neighbour are still
captured, falling back to querying the whole sample when that buffer isn't enough.

# Arguments
- `X` : Sample matrix, rows = observations, columns = coordinates.
- `l` : Row index, in `1:size(X, 1)`, of the query point.
- `coords` : Column indices to restrict the distance computation to (the coalition `v`).
- `k` : Number of nearest neighbours to return (Broto's N_I).
- `rng` : Random number generator used for tie-breaking (keyword, optional).

# Returns
Vector of `k` row indices, in `1:size(X, 1)`, ordered nearest to farthest, excluding `l`.

See the [References](@ref) page for the full citation behind [1].
"""
function _nearest_neighbour_indices(
    X::AbstractMatrix, l::Integer, coords::AbstractVector{<:Integer}, k::Integer;
    rng::AbstractRNG=default_rng(),
)::Vector{Int64}
    n_samples = size(X, 1)
    if k > n_samples - 1
        throw(ArgumentError(
            "Requested $k neighbours (N_I) but only $(n_samples - 1) other sample " *
            "points are available in the dataset.",
        ))
    end

    Xs = permutedims(_standardize_columns(@view X[:, coords]))
    query = Xs[:, l]
    tree = BallTree(Xs)

    # Query a buffer beyond `k` (capped at the full sample) so ties at the k-th nearest
    # neighbour's distance are captured, not just whichever the tree returns first.
    m = min(n_samples, k + 1 + 10)
    idxs, dists = knn(tree, query, m, true)

    self_pos = findfirst(==(l), idxs)
    deleteat!(idxs, self_pos)
    deleteat!(dists, self_pos)

    tie_order = sortperm(collect(zip(dists, randperm(rng, length(dists)))))
    return idxs[tie_order[1:k]]
end

"""
    _nearest_neighbour_pick_freeze(X, Y, Ȳ, s, u; rng=default_rng())::Float64

The V̂^knn_{u,s,PF} estimator of [1], §6.1.2, Eq. (23): the product of the outputs of the
two nearest neighbours (restricted to coordinates `u`) of row `s` of `X`, minus the squared
sample mean of `Y`. Uses only values already present in `(X, Y)` — no new function
evaluations.

The neighbour count is fixed at 2 (see the `N_I` local below) — not a tunable accuracy/cost
knob the way `N_I` is for the (not-yet-implemented) double Monte-Carlo estimator. It's a
structural requirement of the estimator itself: `V_u` is recovered via
`E(E(Y|X_u)²) = E(f(X)f(X^u))` ([1]'s Proposition 2), which holds only because it multiplies
exactly *two* conditionally-independent draws sharing the same conditional mean
`Z = E(Y|X_u)` — the standard `E[Z₁Z₂] = Z²` trick for an unbiased estimator of a squared
mean (`E[Z₁²]` alone would be biased upward by `Var(Z₁)`). Multiplying three or more draws
together would estimate a different, wrong quantity (something like `E[Z³]`), not a more
accurate `V_u` — so this isn't exposed as a keyword.

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

"""
    analyze(X::DataFrame, Y::Vector, n_permutations::Integer; rng=default_rng())::Tuple{Matrix{Float64},Matrix{Float64}}

Estimate Shapley effects from a fixed dataset `(X, Y)` alone — no callable model, no
joint-distribution model — using [1]'s nearest-neighbour "knn" Pick-and-Freeze estimator
(§6.1.2, Eq. 23) combined with [2]'s random-permutation W-aggregation procedure (Eq. 14).
Zero new function evaluations: every conditional-element estimate reuses `Y` values already
present in the dataset.

For each of `n_permutations` random permutations of the factors, a single random reference
row is drawn from `X`; walking the permutation builds nested coalitions
`u_1 ⊂ u_2 ⊂ ... ⊂ u_p = [1:p]`, and each intermediate `V_{u_i}` is estimated via
[`_nearest_neighbour_pick_freeze`](@ref). Consecutive differences accumulate into each factor's
Shapley-effect contribution.

# Arguments
- `X` : Fixed sample of inputs, one row per observation, one column per factor.
- `Y` : Corresponding outputs, `Y[n]` is the already-known output for row `n` of `X`.
- `n_permutations` : Number of random permutations to average over — an accuracy/budget
  knob; more permutations means lower estimator variance.
- `rng` : Random number generator (keyword, optional).

# Returns
Tuple `(Φₙ, Φ²ₙ)`, in the same shape as [`analyze(X::DataFrame, Y::Vector, perms::Matrix)`](@ref)
returns — pass to [`shapley_effects`](@ref) or [`confint`](@ref) as usual.

See the [References](@ref) page for the full citations behind [1] and [2].
"""
function analyze(
    X::DataFrame, Y::Vector, n_permutations::Integer; rng::AbstractRNG=default_rng()
)::Tuple{Matrix{Float64}, Matrix{Float64}}
    Xm = Matrix(X)
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
