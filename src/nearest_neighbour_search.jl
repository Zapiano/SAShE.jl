using NearestNeighbors: BallTree, knn
using Random: AbstractRNG, default_rng, randperm
using Statistics: std

"""
    _standardize_columns(X::AbstractMatrix)::Matrix{Float64}

Z-score each column (coordinate) of `X` by its sample standard deviation.

Coordinates handed to nearest-neighbour search are not assumed to share a common scale —
see the [Dataset-only workflow](@ref) page's Caveats section for why standardized Euclidean
(not Mahalanobis) is the default.

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
