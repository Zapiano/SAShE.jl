using NearestNeighbors: BallTree, knn
using Random: AbstractRNG, default_rng, randperm
using Statistics: std

"""
    _standardize_columns(X::AbstractMatrix)::Matrix{Float64}

Scale each column (coordinate) of `X` by its own sample standard deviation. Note this
divides but does **not** centre — Euclidean distance is translation-invariant, so centring
would be wasted work; the columns are therefore *not* z-scores despite the function's name.

Coordinates handed to nearest-neighbour search are not assumed to share a common scale —
see the [Dataset-only workflow](@ref) page's Caveats section for why standardized Euclidean
(not Mahalanobis) is the default.

A constant column (`σ = 0`) is left unscaled rather than dividing by zero: every row shares
the same value there, so it contributes exactly zero to every pairwise distance regardless
of what it's divided by — scaling it is a no-op either way, so skipping the division just
avoids a spurious `NaN`.

`NaN` columns are caught by the caller, which knows the factor indices — see
[`_nearest_neighbour_indices`](@ref).

# Arguments
- `X` : Sample matrix, rows = observations, columns = coordinates.

# Returns
Matrix of the same size as `X`, with each column divided by its own sample standard
deviation (constant columns left as-is).
"""
function _standardize_columns(X::AbstractMatrix)::Matrix{Float64}
    σ = std(X; dims=1)
    σ[σ .== 0] .= 1.0
    return X ./ σ
end

"""
    _nearest_neighbour_indices(X, l, coords, k; rng=default_rng())::Vector{Int64}

Indices, in `1:size(X, 1)`, of the `k` nearest neighbours of row `l` of `X`, restricted to
the columns in `coords`, **excluding `l` itself**.

This is [1] §6's `k^v_N(l, n)` primitive — the n-th nearest neighbour of sample point `l`
restricted to coordinate subset `v` — but **offset by one**: [1] searches the full sample
*including* `l`, so its `k^v_N(l, 1)` is `l` at distance zero (see [1]'s Remark 20, and
[`_nearest_neighbour_pick_freeze`](@ref)'s docstring, which relies on exactly that). This
function's returned index `1` is therefore [1]'s `k^v_N(l, 2)`. Callers translate: asking for
`k = 1` here gives the single neighbour [1]'s `N_I = 2` pick-and-freeze estimator needs, so
`k` is *not* Broto's `N_I`.

Distance is standardized Euclidean (see [`_standardize_columns`](@ref)). Ties at the k-th
nearest neighbour's distance are broken uniformly at random ([1]'s Assumption 7), rather
than by whichever order the search happens to return — this matters for discrete numeric
coordinates, where exact ties are possible even without categorical data.

To keep that tie-breaking exact without paying for it when it isn't needed, the search starts
at `k + 2` neighbours and grows (doubling, up to the whole sample) only while a tie actually
straddles the k-th boundary — i.e. while `dists[k] == dists[k+1]` — or while `l` itself has
been pushed out of the returned set by exact duplicates. [1] notes ties are impossible when
the inputs are absolutely continuous, so the common case never grows past the first small
query. Querying the whole sample unconditionally instead costs `O(N log N)` rather than
`O(log N)` — two to three orders of magnitude on the query at `N = 8000` — but only ~1.5-1.7×
end-to-end, because rebuilding the `BallTree` on every call dominates either way (see issue
#24). Exact ratios are machine-dependent; treat them as indicative.

A `NaN` in any queried coordinate is rejected here rather than allowed to propagate: it would
otherwise slip past [`_standardize_columns`](@ref)'s constant-column guard (`std` of such a
column is `NaN`, and `NaN == 0` is false) and silently turn every distance into `NaN`.

# Arguments
- `X` : Sample matrix, rows = observations, columns = coordinates.
- `l` : Row index, in `1:size(X, 1)`, of the query point.
- `coords` : Column indices to restrict the distance computation to (the coalition `v`).
- `k` : Number of neighbours to return, excluding `l` — see the offset note above; this is
  *not* [1]'s `N_I`.
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
            "Requested $k neighbours but only $(n_samples - 1) other sample " *
            "points are available in the dataset.",
        ))
    end

    # Reported against the caller's own factor numbering, not the position within `coords`.
    bad = findfirst(j -> any(isnan, @view X[:, j]), coords)
    isnothing(bad) || throw(
        ArgumentError(
            "Factor (column) $(coords[bad]) contains NaN, so nearest-neighbour distances " *
            "are undefined. Remove or impute the missing values before analyzing.",
        ),
    )

    Xs = permutedims(_standardize_columns(@view X[:, coords]))
    query = Xs[:, l]
    tree = BallTree(Xs)

    # `k` neighbours, plus `l` itself, plus one lookahead to detect a tie at the boundary.
    m = min(n_samples, k + 2)
    while true
        idxs, dists = knn(tree, query, m, true)

        self_pos = findfirst(==(l), idxs)
        if self_pos !== nothing
            deleteat!(idxs, self_pos)
            deleteat!(dists, self_pos)
        end

        # The k-th boundary is unambiguous once the whole sample has been seen, or once `l`
        # is accounted for and the k-th and (k+1)-th distances differ — then no tie group
        # straddles the cut and the first `k` are the answer whatever the tie-break says.
        if m == n_samples ||
            (self_pos !== nothing && length(dists) > k && dists[k] != dists[k + 1])
            tie_order = sortperm(collect(zip(dists, randperm(rng, length(dists)))))
            return idxs[tie_order[1:k]]
        end

        m = min(n_samples, 2m)
    end
end
