using Random: AbstractRNG, default_rng
using Statistics: mean, var

"""
    MixModel(func::Function)

Wraps a callable model for use with [`analyze`](@ref) — the "what you have" container for
the case where a model is callable but its inputs' joint distribution isn't known well
enough to sample fresh, valid points from, only an existing dataset. Carries no sample data
itself: pair it with a [`MixPickAndFreezeSample`](@ref) or
[`MixDoubleMonteCarloSample`](@ref), built independently, to run and analyze `func`.

# Arguments
- `func` : A function that accepts a vector of factor values and returns a scalar.

# Examples
```julia
model = MixModel(func)
sample = MixPickAndFreezeSample(X, Y, 5000)
Φₙ, Φ²ₙ, Yₙ = analyze(model, sample)
```
"""
struct MixModel
    func::Function
end

function Base.:show(io::IO, m::MixModel)
    return print(io, "MixModel(", m.func, ")")
end

"""
    _check_mix_dataset_consistency(func, X, Y; rng=default_rng(), n_checks=10)

Spot-checks that `Y` really is `func` evaluated at `X`, row for row. Both mix estimators
call `func` at the reference row itself as their first term (see
[`_nearest_neighbour_mix_pick_freeze`](@ref)), and `analyze`'s terminal `W = var(sample.Y)`
is computed from `sample.Y` directly rather than from `func`, while every other `W` genuinely
calls `func`. A mismatch between `Y` and `func` biases `var(sample.Y)` against the
`func`-derived intermediate `W`s, and thus the last step of every permutation's telescoping
sum, so this checks a random subset of rows (not all — `func` may be expensive) and raises
an error on the first mismatch rather than returning silently wrong effects.

# Arguments
- `func` : The callable model.
- `X` : Sample matrix, rows = observations, columns = coordinates.
- `Y` : Corresponding outputs to check against `func`.
- `rng` : Random number generator used to pick which rows to check (keyword, optional).
- `n_checks` : Number of rows to check (keyword, optional).
"""
function _check_mix_dataset_consistency(
    func, X::AbstractMatrix, Y::AbstractVector;
    rng::AbstractRNG=default_rng(), n_checks::Integer=10,
)
    n_samples = size(X, 1)
    for i ∈ rand(rng, 1:n_samples, min(n_checks, n_samples))
        fx = func(X[i, :])
        isapprox(fx, Y[i]; rtol=1e-6, atol=1e-8) || throw(
            ArgumentError(
                "`Y[$i]` ($(Y[i])) does not match `func` at that row ($fx). MixModel's " *
                "estimators assume `Y[n] == func(X[n, :])` exactly for every row — see " *
                "MixPickAndFreezeSample's docstring.",
            ),
        )
    end
    return nothing
end

"""
    analyze(model::MixModel, sample::MixPickAndFreezeSample; rng=default_rng(), n_checks=10)::Tuple{Matrix{Float64},Matrix{Float64},Vector{Float64}}
    analyze(model::MixModel, sample::MixDoubleMonteCarloSample; rng=default_rng(), n_checks=10)::Tuple{Matrix{Float64},Matrix{Float64},Vector{Float64}}

Estimate Shapley effects by calling `model.func` at nearest-neighbour-selected points of
`sample`'s existing dataset — no known joint distribution to sample fresh inputs from. Which
estimator runs is determined entirely by `sample`'s type, matching
[`analyze(m::CallableModel, s::CallablePickAndFreezeSample)`](@ref):

- [`MixPickAndFreezeSample`](@ref) → [1]'s "mix" Pick-and-Freeze estimator (§6.1.2, Eq. 22)
  via [`_nearest_neighbour_mix_pick_freeze`](@ref).
- [`MixDoubleMonteCarloSample`](@ref) → [1]'s "mix" double Monte-Carlo estimator (§6.1.1,
  Eq. 17) via [`_nearest_neighbour_mix_double_monte_carlo`](@ref).

Both combine with the random-permutation W-aggregation procedure from [2], its Eq. 12, in the
form [1] states it in §4.2, Eq. 14, unnormalized — `Φₙ` summing to `Var(Y)` rather than 1,
matching [`DataModel`](@ref) — see
[Deliberate deviations](@ref). For each permutation `sample` was built with, walking it grows
nested coalitions `u_1 ⊂ u_2 ⊂ ... ⊂ u_p = [1:p]` from the permutation's reference row, and
each intermediate `W_{u_i}` is estimated by calling `model.func` at synthetic points built
from nearest-neighbour lookups into the dataset — not by reusing `Y`, which is the "mix" vs.
"knn" distinction ([`DataModel`](@ref) implements "knn"). The terminal `W = var(sample.Y)`
comes from `sample.Y` directly rather than from `model.func`, though — see
[`_check_mix_dataset_consistency`](@ref) for the assumption this relies on and the check
`analyze` runs for it.

# Arguments
- `model` : The model to run, wrapped in a [`MixModel`](@ref).
- `sample` : The dataset and pre-drawn randomness to walk, wrapped in a
  [`MixPickAndFreezeSample`](@ref) or a [`MixDoubleMonteCarloSample`](@ref).
- `rng` : Random number generator used only for the nearest-neighbour search's tie-breaking
  (keyword, optional) — see [`MixPickAndFreezeSample`](@ref) for why this isn't part of
  `sample` itself.
- `n_checks` : Number of rows [`_check_mix_dataset_consistency`](@ref) spot-checks before
  the walk starts (keyword, optional). Each check costs one `func` call — for an expensive
  `func`, lower this (or pass `0` to skip the check entirely) once you trust `sample.Y`.

# Returns
Tuple `(Φₙ, Φ²ₙ, Yₙ)`, matching
[`analyze(m::CallableModel, s::CallablePickAndFreezeSample)`](@ref)'s arity — pass `Φₙ`/
`Φ²ₙ` to [`shapley_effects`](@ref) or [`confint`](@ref) as usual.

    - Φₙ, Φ²ₙ : As above.
    - Yₙ : Every value `model.func` was called at during the walk, in call order (including
      the "recomputed" reference-row calls noted in [`_nearest_neighbour_mix_pick_freeze`](@ref)
      and [`_nearest_neighbour_mix_double_monte_carlo`](@ref)'s docstrings). Unlike
      [`analyze(m::CallableModel, s::CallablePickAndFreezeSample)`](@ref)'s `Yₙ`, this is
      *not* one value per row of a fixed sample table — `MixModel` builds synthetic points
      adaptively via nearest-neighbour search rather than evaluating a pre-built table, so
      there is no table row for each entry to align with. It's provided for the same reason:
      diagnostics and sanity checks, not for indexing back into `sample.X`.

See the [References](@ref) page for the full citations behind [1] and [2].
"""
function analyze(
    model::MixModel, sample::MixPickAndFreezeSample;
    rng::AbstractRNG=default_rng(), n_checks::Integer=10,
)::Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}
    Xm = Matrix(sample.X)
    func = model.func
    _check_mix_dataset_consistency(func, Xm, sample.Y; rng=rng, n_checks=n_checks)
    perms = sample.permutations
    refs = sample.references
    n_permutations, n_factors = size(perms)
    Ȳ = mean(sample.Y)
    var_y = var(sample.Y)

    Φₙ_increments = zeros(n_factors, n_permutations)
    Φₙ²_increments = zeros(n_factors, n_permutations)
    # Exact final length: two `func` calls per non-terminal coalition step (the terminal step
    # reads `var_y` and calls nothing), for every permutation.
    Yₙ = Vector{Float64}(undef, n_permutations * (n_factors - 1) * 2)
    n_called = 0

    for m ∈ 1:n_permutations
        σ = @view perms[m, :]
        s = refs[m]

        prevW = 0.0
        for i ∈ 1:n_factors
            # The coalition after `i` steps is exactly the permutation's first `i` entries,
            # so it needs no buffer of its own.
            u = @view σ[1:i]
            W = if i == n_factors
                var_y
            else
                W, calls = _nearest_neighbour_mix_pick_freeze(func, Xm, Ȳ, s, u; rng=rng)
                copyto!(Yₙ, n_called + 1, calls)
                n_called += length(calls)
                W
            end

            Δ = W - prevW
            Φₙ_increments[σ[i], m] = Δ / n_permutations
            Φₙ²_increments[σ[i], m] = Δ^2 / n_permutations

            prevW = W
        end
    end

    # `Yₙ` is `undef`-initialized, so a short fill would return garbage rather than error.
    @assert n_called == length(Yₙ) "Yₙ was sized for $(length(Yₙ)) calls but $n_called were made"

    return Φₙ_increments, Φₙ²_increments, Yₙ
end
function analyze(
    model::MixModel, sample::MixDoubleMonteCarloSample;
    rng::AbstractRNG=default_rng(), n_checks::Integer=10,
)::Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}
    Xm = Matrix(sample.X)
    func = model.func
    _check_mix_dataset_consistency(func, Xm, sample.Y; rng=rng, n_checks=n_checks)
    perms = sample.permutations
    refs = sample.references
    N_I = sample.N_I
    n_permutations, n_factors = size(perms)
    var_y = var(sample.Y)

    Φₙ_increments = zeros(n_factors, n_permutations)
    Φₙ²_increments = zeros(n_factors, n_permutations)
    # Exact final length: `N_I` `func` calls per non-terminal coalition step (the terminal
    # step reads `var_y` and calls nothing), for every permutation.
    Yₙ = Vector{Float64}(undef, n_permutations * (n_factors - 1) * N_I)
    n_called = 0

    for m ∈ 1:n_permutations
        σ = @view perms[m, :]
        s = refs[m]

        prevW = 0.0
        for i ∈ 1:n_factors
            # The coalition after `i` steps is exactly the permutation's first `i` entries,
            # so it needs no buffer of its own.
            u = @view σ[1:i]
            W = if i == n_factors
                var_y
            else
                W, calls = _nearest_neighbour_mix_double_monte_carlo(func, Xm, s, u, N_I; rng=rng)
                copyto!(Yₙ, n_called + 1, calls)
                n_called += length(calls)
                W
            end

            Δ = W - prevW
            Φₙ_increments[σ[i], m] = Δ / n_permutations
            Φₙ²_increments[σ[i], m] = Δ^2 / n_permutations

            prevW = W
        end
    end

    # `Yₙ` is `undef`-initialized, so a short fill would return garbage rather than error.
    @assert n_called == length(Yₙ) "Yₙ was sized for $(length(Yₙ)) calls but $n_called were made"

    return Φₙ_increments, Φₙ²_increments, Yₙ
end

"""
    _nearest_neighbour_mix_pick_freeze(func, X, Ȳ, s, u; rng=default_rng())::Tuple{Float64,Vector{Float64}}

The V̂^mix_{u,s,PF} estimator of [1], §6.1.2, Eq. (22): `func(x_s) * func(hybrid) - Ȳ²`, where
`x_s` is the reference row and `hybrid` shares `x_s`'s coordinates in `u` but takes its
coordinates outside `u` from `x_s`'s nearest neighbour (restricted to `u`) — the "mix"
counterpart of [`_nearest_neighbour_pick_freeze`](@ref), which reuses `Y[s]`/`Y[NN₁(s)]`
instead of calling `func`.

As with [`_nearest_neighbour_pick_freeze`](@ref), [1]'s neighbour-index definition searches
the full sample including `s` itself, so the first term is `func` at the reference point
verbatim (not a synthetic point) — see [`_nearest_neighbour_indices`](@ref)'s docstring for
the offset convention this relies on. Note `func(x_s)` here recomputes a value already known
exactly as the dataset's own `Y[s]`; this follows [1]'s Eq. (22) literally rather than
substituting it — see issue #29 for the optimization this leaves on the table.

# Arguments
- `func` : The callable model.
- `X` : Sample matrix, rows = observations, columns = coordinates.
- `Ȳ` : Sample mean of the dataset's `Y` (passed in so callers don't recompute it every call).
- `s` : Row index, in `1:size(X, 1)`, of the reference point.
- `u` : Coalition (column indices), nonempty and a proper subset of the factors.
- `rng` : Random number generator used for tie-breaking (keyword, optional).

# Returns
A `Tuple{Float64,Vector{Float64}}`: the estimate of `V_u = Var(E(Y | X_u))`, and the two
`func` values (`[func(x_s), func(hybrid)]`) it was computed from — the latter is for
[`analyze`](@ref)'s `Yₙ` return value, not part of the estimator itself.

See the [References](@ref) page for the full citation behind [1].
"""
function _nearest_neighbour_mix_pick_freeze(
    func, X::AbstractMatrix, Ȳ::Real, s::Integer, u::AbstractVector{<:Integer};
    rng::AbstractRNG=default_rng(),
)::Tuple{Float64, Vector{Float64}}
    notU = setdiff(1:size(X, 2), u)
    neighbour = only(_nearest_neighbour_indices(X, s, u, 1; rng=rng))

    x_s = X[s, :]
    hybrid = copy(x_s)
    hybrid[notU] .= @view X[neighbour, notU]

    f_x_s, f_hybrid = func(x_s), func(hybrid)
    return f_x_s * f_hybrid - Ȳ^2, [f_x_s, f_hybrid]
end

"""
    _nearest_neighbour_mix_double_monte_carlo(func, X, s, u, N_I; rng=default_rng())::Tuple{Float64,Vector{Float64}}

The Ê^mix_{u,s,MC} estimator of [1], §6.1.1, Eq. (17): the Bessel-corrected sample variance
of `func` evaluated at `N_I` synthetic points sharing the reference row's coordinates outside
`u` (held fixed) but varying inside `u`, taken from `N_I` nearest neighbours of the reference
row restricted to the coordinates *outside* `u` — the "mix" counterpart of double Monte
Carlo's inner-sample-variance step, approximating draws from `Y`'s conditional distribution
given `X₋ᵤ` via real dataset rows that are close on those coordinates, rather than drawing
from a known one.

As in [`_nearest_neighbour_mix_pick_freeze`](@ref), the first of the `N_I` neighbours is the
reference row itself ([1]'s neighbour-index definition includes it, at distance zero), so the
first replicate is `func` at the reference point verbatim and only `N_I - 1` external
neighbours are queried.

# Arguments
- `func` : The callable model.
- `X` : Sample matrix, rows = observations, columns = coordinates.
- `s` : Row index, in `1:size(X, 1)`, of the reference point.
- `u` : Coalition (column indices) to vary, nonempty and a proper subset of the factors —
  the complement, held fixed, is what the neighbour search matches on.
- `N_I` : Number of replicate evaluations to build the sample variance from; must be `≥ 2`.
- `rng` : Random number generator used for tie-breaking (keyword, optional).

# Returns
A `Tuple{Float64,Vector{Float64}}`: the estimate of `c(u) = E(Var(Y | X₋ᵤ))`, and the `N_I`
`func` values it was computed from — the latter is for [`analyze`](@ref)'s `Yₙ` return
value, not part of the estimator itself.

See the [References](@ref) page for the full citation behind [1].
"""
function _nearest_neighbour_mix_double_monte_carlo(
    func, X::AbstractMatrix, s::Integer, u::AbstractVector{<:Integer}, N_I::Integer;
    rng::AbstractRNG=default_rng(),
)::Tuple{Float64, Vector{Float64}}
    notU = setdiff(1:size(X, 2), u)
    neighbours = _nearest_neighbour_indices(X, s, notU, N_I - 1; rng=rng)

    x_s = X[s, :]
    f_vals = zeros(N_I)
    f_vals[1] = func(x_s)
    for (i, idx) ∈ enumerate(neighbours)
        hybrid = copy(x_s)
        hybrid[u] .= @view X[idx, u]
        f_vals[i + 1] = func(hybrid)
    end

    return var(f_vals), f_vals
end
