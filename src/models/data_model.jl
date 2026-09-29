using Random: AbstractRNG, default_rng, randperm
using Statistics: mean, var

"""
    DataModel(X::DataFrame, Y::Vector)

Container for the "fixed dataset, no callable model" case: wraps the `(X, Y)` pair
analyzed by [`analyze`](@ref).

# Arguments
- `X` : Fixed sample of inputs, one row per observation, one column per factor.
- `Y` : Corresponding outputs, `Y[n]` is the already-known output for row `n` of `X`.
"""
struct DataModel
    X::DataFrame
    Y::Vector

    function DataModel(X::DataFrame, Y::Vector)
        size(X, 1) == length(Y) || throw(
            ArgumentError(
                "X and Y must have the same number of rows, got $(size(X, 1)) and $(length(Y))",
            ),
        )
        return new(X, Y)
    end
end

"""
    analyze(model::DataModel, n_permutations::Integer, ::PickAndFreeze; rng=default_rng())::Tuple{Matrix{Float64},Matrix{Float64}}

Estimate Shapley effects from a fixed dataset `(X, Y)` alone — no callable model, no
joint-distribution model — via [1]'s nearest-neighbour "knn" Pick-and-Freeze estimator
(§6.1.2, Eq. 23) combined with the random-permutation W-aggregation procedure originally
from [2], its Eq. 12, in the form [1] states it in §4.2, Eq. 14 — unnormalized, though:
[1]'s Eq. (14) divides by `Var(Y)` to return effects summing to 1, while this returns
`Φₙ` summing to `Var(Y)` instead. Note `sum(Φ) == var(Y)` holds
**exactly** here, by construction (the walk's terminal step is `var(Y)` itself, not a value
derived from the estimator) — it's not a check on whether the nearest-neighbour estimator
is working, only on whether the aggregation is wired correctly. Zero new function
evaluations: every conditional-element estimate reuses `Y` values already present in the
dataset.

For each of `n_permutations` random permutations of the factors, a single random reference
row is drawn from `X`; walking the permutation builds nested coalitions
`u_1 ⊂ u_2 ⊂ ... ⊂ u_p = [1:p]`, and each intermediate `V_{u_i}` is estimated via
[`_nearest_neighbour_pick_freeze`](@ref). Consecutive differences accumulate into each factor's
Shapley-effect contribution.

`DataModel` also implements [`DoubleMonteCarlo`](@ref) — see
[`analyze(model::DataModel, n_permutations::Integer, ::DoubleMonteCarlo)`](@ref). An
`EstimationMethod` value is required explicitly here; there is no default, see
[Deliberate deviations](@ref).

# Arguments
- `model` : The dataset to analyze, wrapped in a [`DataModel`](@ref).
- `n_permutations` : Number of random permutations to average over — an accuracy/budget
  knob; more permutations means lower estimator variance.
- `rng` : Random number generator (keyword, optional).

# Returns
Tuple `(Φₙ, Φ²ₙ)`, in the same shape as
[`analyze(m::CallableModel, s::CallablePickAndFreezeSample)`](@ref) returns — pass to
[`shapley_effects`](@ref) or [`confint`](@ref) as usual.
"""
function analyze(
    model::DataModel, n_permutations::Integer, ::PickAndFreeze; rng::AbstractRNG=default_rng()
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

The V̂^knn_{u,s,PF} estimator of [1], §6.1.2, Eq. (23): the product of `Y[s]` (the query
point's own, exact output) and the output of its nearest neighbour (restricted to
coordinates `u`), minus the squared sample mean of `Y`. Uses only values already present in
`(X, Y)` — no new function evaluations.

[1]'s own neighbour-index definition (§6, before Eq. 23) searches the *full* sample
including the query point itself, so its first "nearest neighbour" of `s` is `s` at distance
zero — i.e. Eq. (23) is `Y[s] · Y[NN₁(s)]`, not the product of two neighbours *other* than
`s`. Pairing `s` with one neighbour (rather than two neighbours, neither of them `s`) is
what this implements; it also uses one fewer neighbour query per call.

The neighbour count is fixed, not a tunable accuracy/cost knob: one neighbour besides `s`
itself, which is [1]'s `N_I = 2` (§6.1.2 opens by fixing `N_I` at 2 for the pick-and-freeze
estimators). It can't be raised: `V_u = Var(E(Y|X_u))` is recovered via the `E[Z₁Z₂] = Z²`
identity, which only holds for exactly *two* conditionally-independent draws sharing the
same conditional mean `Z = E(Y|X_u)` — multiplying three or more draws together would
estimate a different, wrong quantity, not a more accurate `V_u`.

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
"""
function _nearest_neighbour_pick_freeze(
    X::AbstractMatrix, Y::AbstractVector{<:Real}, Ȳ::Real, s::Integer,
    u::AbstractVector{<:Integer}; rng::AbstractRNG=default_rng(),
)::Float64
    # One neighbour besides `s` itself, i.e. [1]'s N_I = 2 — fixed by the estimator's own
    # structure, not a tunable accuracy knob. See this function's docstring, and
    # `_nearest_neighbour_indices`'s for why `k` here is not [1]'s `N_I`.
    k = 1
    idx = _nearest_neighbour_indices(X, s, u, k; rng=rng)
    return Y[s] * Y[idx[1]] - Ȳ^2
end

"""
    analyze(model::DataModel, n_permutations::Integer, ::DoubleMonteCarlo; N_I=3, rng=default_rng())::Tuple{Matrix{Float64},Matrix{Float64}}

The [`DoubleMonteCarlo`](@ref) counterpart of
[`analyze(model::DataModel, n_permutations::Integer, ::PickAndFreeze)`](@ref): estimates
Shapley effects from a fixed dataset `(X, Y)` alone via [1]'s "knn" double Monte-Carlo
estimator (§6.1.1, Eq. 18), combined with the same random-permutation W-aggregation
procedure. Zero new function evaluations, exactly as the Pick-and-Freeze method — every
coalition's intermediate value reuses `Y` values already present in the dataset, this time via
[`_nearest_neighbour_double_monte_carlo`](@ref) rather than
[`_nearest_neighbour_pick_freeze`](@ref).

# Arguments
- `model` : The dataset to analyze, wrapped in a [`DataModel`](@ref).
- `n_permutations` : Number of random permutations to average over — an accuracy/budget
  knob; more permutations means lower estimator variance.
- `N_I` : Number of nearest neighbours (including the reference row itself) each coalition's
  sample variance is computed from (keyword, optional, default `3`, [1]'s own example choice
  in §6.1.1); must be `≥ 2`. Unlike Pick-and-Freeze's fixed `N_I = 2`, this is a genuine
  accuracy/cost knob here — see [`_nearest_neighbour_double_monte_carlo`](@ref)'s docstring for
  why.
- `rng` : Random number generator (keyword, optional).

# Returns
Tuple `(Φₙ, Φ²ₙ)`, in the same shape as
[`analyze(model::DataModel, n_permutations::Integer, ::PickAndFreeze)`](@ref) returns — pass
to [`shapley_effects`](@ref) or [`confint`](@ref) as usual.
"""
function analyze(
    model::DataModel, n_permutations::Integer, ::DoubleMonteCarlo;
    N_I::Integer=3, rng::AbstractRNG=default_rng(),
)::Tuple{Matrix{Float64}, Matrix{Float64}}
    N_I >= 2 ||
        throw(ArgumentError("N_I must be ≥ 2 to estimate a sample variance, got $N_I"))

    Xm = Matrix(model.X)
    Y = model.Y
    n_samples, n_factors = size(Xm)
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
                _nearest_neighbour_double_monte_carlo(Xm, Y, s, u, N_I; rng=rng)
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
    _nearest_neighbour_double_monte_carlo(X, Y, s, u, N_I; rng=default_rng())::Float64

The Ê^knn_{u,s,MC} estimator of [1], §6.1.1, Eq. (18): the Bessel-corrected sample variance of
`Y` at `N_I` nearest neighbours of the reference row `s`, restricted to the coordinates
*outside* `u` (held fixed) — the "knn" counterpart of
[`_nearest_neighbour_mix_double_monte_carlo`](@ref), which reuses `Y` values already present in
the dataset instead of calling `func` at synthetic points.

As in [`_nearest_neighbour_pick_freeze`](@ref), [1]'s neighbour-index definition includes the
reference row itself, at distance zero, so the first of the `N_I` values is `Y[s]` verbatim and
only `N_I - 1` external neighbours are actually queried.

Unlike [`_nearest_neighbour_pick_freeze`](@ref)'s `N_I`, fixed at 2 by that estimator's own
`E[Z₁Z₂] = Z²` structure, this estimator's `N_I` is a genuine accuracy/cost knob: Eq. (18) is a
sample variance over `N_I` draws, and a sample variance remains a valid (if noisier) estimator
of the same quantity for any `N_I ≥ 2` — there's no algebraic identity here that breaks for
`N_I > 2`, only more neighbours traded for more accuracy. [1] suggests `N_I = 3` as an example
(§6.1.1), which this package also uses as the default.

# Arguments
- `X` : Sample matrix, rows = observations, columns = coordinates.
- `Y` : Corresponding outputs, `Y[n]` is the already-known output for row `n` of `X`.
- `s` : Row index, in `1:size(X, 1)`, of the reference point.
- `u` : Coalition (column indices) to vary, nonempty and a proper subset of the factors — the
  complement, held fixed, is what the neighbour search matches on.
- `N_I` : Number of values (including `Y[s]` itself) to build the sample variance from; must
  be `≥ 2`.
- `rng` : Random number generator used for tie-breaking (keyword, optional).

# Returns
A single `Float64`: the estimate of `c(u) = E(Var(Y | X₋ᵤ))`.
"""
function _nearest_neighbour_double_monte_carlo(
    X::AbstractMatrix, Y::AbstractVector{<:Real}, s::Integer, u::AbstractVector{<:Integer},
    N_I::Integer; rng::AbstractRNG=default_rng(),
)::Float64
    notU = setdiff(1:size(X, 2), u)
    neighbours = _nearest_neighbour_indices(X, s, notU, N_I - 1; rng=rng)

    y_vals = Vector{Float64}(undef, N_I)
    y_vals[1] = Y[s]
    y_vals[2:end] .= @view Y[neighbours]

    return var(y_vals)
end
