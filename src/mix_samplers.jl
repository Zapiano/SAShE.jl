using DataFrames
using Random: AbstractRNG, default_rng

"""
    MixPickAndFreezeSample(X::DataFrame, Y::Vector, n_permutations::Integer; rng=default_rng())
    MixPickAndFreezeSample(X::DataFrame, Y::Vector, permutations::Matrix{Int64}, references::Vector{Int64})

The existing dataset (`X`, `Y`) plus the pre-drawn randomness [`analyze`](@ref) walks a
[`MixModel`](@ref) over: one random permutation and one random reference row per coalition
walk, matching [1]'s random-permutation W-aggregation procedure (`N_u = N_O = 1` — one
reference row per permutation, reused for every coalition step within it).

Building this ahead of time — rather than drawing randomness inside `analyze` — matches how
[`CallablePickAndFreezeSample`](@ref)/[`CallableDoubleMonteCarloSample`](@ref) work: everything except the
model's own function calls is fixed once the sample is built, so re-running `analyze` on the
same sample is reproducible without needing to pass an `rng` to it. (`analyze` does still take
an `rng` keyword of its own — only for the nearest-neighbour search's tie-breaking, which
happens during the walk and only matters when the dataset has exact ties; see
[`_nearest_neighbour_indices`](@ref).)

# Arguments
- `X` : Existing sample of inputs, one row per observation, one column per factor.
- `Y` : Corresponding outputs. Must satisfy `Y[n] == func(X[n, :])` exactly for every row —
  `analyze`'s terminal `W` (the coalition of all factors) is `var(sample.Y)` rather than a
  `func` call, while every other `W` genuinely calls `func` (see
  [`_nearest_neighbour_mix_pick_freeze`](@ref)); a systematic disagreement between `Y` and
  `func` (noisy measurements, or `Y` from a different function than the one passed to
  [`MixModel`](@ref)) biases that one step against the rest. `analyze` spot-checks this and
  raises an error if it doesn't hold.
- `n_permutations` : Number of random permutations to draw — an accuracy/budget knob; more
  permutations means lower estimator variance.
- `permutations` : Permutation matrix to use instead of generating a new one (`n_permutations
  × n_factors`, one row per permutation).
- `references` : Reference row index into `X` to use for each permutation, instead of drawing
  new ones (length `n_permutations`, values in `1:size(X, 1)`).
- `rng` : Random number generator (keyword, optional; first form only).

See the [References](@ref) page for the full citation behind [1].
"""
struct MixPickAndFreezeSample
    X::DataFrame
    Y::Vector
    permutations::Matrix{Int64}
    references::Vector{Int64}

    function MixPickAndFreezeSample(
        X::DataFrame, Y::Vector, permutations::Matrix{Int64}, references::Vector{Int64}
    )
        _validate_mix_sample(X, Y, permutations, references)
        return new(X, Y, permutations, references)
    end

    function MixPickAndFreezeSample(
        X::DataFrame, Y::Vector, n_permutations::Integer; rng::AbstractRNG=default_rng()
    )
        n_factors = size(X, 2)
        permutations = generate_permutations(n_permutations, n_factors; rng=rng)
        references = rand(rng, 1:size(X, 1), n_permutations)
        return MixPickAndFreezeSample(X, Y, permutations, references)
    end
end

"""
    MixDoubleMonteCarloSample(X::DataFrame, Y::Vector, n_permutations::Integer; N_I=3, rng=default_rng())
    MixDoubleMonteCarloSample(X::DataFrame, Y::Vector, permutations::Matrix{Int64}, references::Vector{Int64}, N_I::Integer)

The existing dataset (`X`, `Y`) plus the pre-drawn randomness [`analyze`](@ref) walks a
[`MixModel`](@ref) over with the double Monte-Carlo estimator — see
[`MixPickAndFreezeSample`](@ref) for why the permutations and reference rows are drawn ahead
of time rather than inside `analyze`.

# Arguments
- `X` : Existing sample of inputs, one row per observation, one column per factor.
- `Y` : Corresponding outputs. Must satisfy `Y[n] == func(X[n, :])` exactly for every row —
  see [`MixPickAndFreezeSample`](@ref) for why (the same terminal-`W` reasoning applies
  here), and note `analyze` spot-checks this.
- `n_permutations` : Number of random permutations to draw — an accuracy/budget knob.
- `N_I` : Number of nearest-neighbour points to build each coalition step's sample variance
  from; must be `≥ 2`. A genuine accuracy/cost knob, unlike pick-and-freeze's structurally
  fixed neighbour count — matches [`CallableDoubleMonteCarloSample`](@ref)'s default of `3`.
- `permutations` : Permutation matrix to use instead of generating a new one.
- `references` : Reference row index into `X` to use for each permutation, instead of drawing
  new ones.
- `rng` : Random number generator (keyword, optional; first form only).

See the [References](@ref) page for the full citation behind [1].
"""
struct MixDoubleMonteCarloSample
    X::DataFrame
    Y::Vector
    permutations::Matrix{Int64}
    references::Vector{Int64}
    N_I::Int64

    function MixDoubleMonteCarloSample(
        X::DataFrame, Y::Vector, permutations::Matrix{Int64}, references::Vector{Int64},
        N_I::Integer,
    )
        N_I >= 2 ||
            throw(ArgumentError("N_I must be ≥ 2 to estimate a sample variance, got $N_I"))
        _validate_mix_sample(X, Y, permutations, references)
        return new(X, Y, permutations, references, N_I)
    end

    function MixDoubleMonteCarloSample(
        X::DataFrame, Y::Vector, n_permutations::Integer;
        N_I::Integer=3, rng::AbstractRNG=default_rng(),
    )
        n_factors = size(X, 2)
        permutations = generate_permutations(n_permutations, n_factors; rng=rng)
        references = rand(rng, 1:size(X, 1), n_permutations)
        return MixDoubleMonteCarloSample(X, Y, permutations, references, N_I)
    end
end

"""
    _validate_mix_sample(X, Y, permutations, references)

Shape and content checks shared by [`MixPickAndFreezeSample`](@ref) and
[`MixDoubleMonteCarloSample`](@ref): `X`/`Y` row counts match, `permutations` has one column
per factor and one row per reference and each row is a genuine permutation of `1:n_factors`
(not just the right shape — a row like `[1, 1, 1]` would otherwise silently overwrite the
same `Φₙ_increments` slot instead of erroring), and every `references` entry is a valid row
of `X`.
"""
function _validate_mix_sample(
    X::DataFrame, Y::Vector, permutations::Matrix{Int64}, references::Vector{Int64}
)
    n_samples, n_factors = size(X)
    n_permutations = size(permutations, 1)
    errors::Vector{String} = []
    n_samples == length(Y) ||
        push!(errors, "`X` and `Y` must have the same number of rows")
    size(permutations, 2) == n_factors ||
        push!(errors, "`permutations` must have one column per factor in `X`")
    if size(permutations, 2) == n_factors &&
        !all(r -> sort(collect(r)) == 1:n_factors, eachrow(permutations))
        push!(errors, "every row of `permutations` must be a permutation of 1:$n_factors")
    end
    length(references) == n_permutations ||
        push!(errors, "`references` must have one entry per row of `permutations`")
    isempty(references) || (1 <= minimum(references) && maximum(references) <= n_samples) ||
        push!(errors, "`references` must index rows of `X` (i.e. lie in 1:$n_samples)")
    return isempty(errors) ? nothing : throw(ArgumentError(join(errors, "\n")))
end
