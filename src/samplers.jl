using DataFrames
using Random
using Distributions
import QuasiMonteCarlo as QMC

"""
    PickAndFreezeSample(factor_names::Union{Vector{String},Vector{Symbol}}, n_samples::Int64, factor_dist::Vector{<:Distribution}, strategy::SamplingStrategy; rng=default_rng())
    PickAndFreezeSample(A::DataFrame, B::DataFrame; conditional_sampler=nothing, rng=default_rng())
    PickAndFreezeSample(A::DataFrame, B::DataFrame, permutations::Matrix{Int64}; conditional_sampler=nothing)
    PickAndFreezeSample(samples::DataFrame, permutations::Matrix{Int64})

SAShE samples (`X`) and permutations (`π`), the pick-and-freeze sample table [`analyze`](@ref)
runs a [`CallableModel`](@ref) over.

Pass `conditional_sampler` when some factors are dependent: after the pick-freeze samples
are built, every non-base row has its resampled factors redrawn conditional on the frozen
ones. See [`_conditionally_resample!`](@ref) for the expected signature.

The last form wraps an already pick-freeze-shaped `samples`/`permutations` pair directly —
e.g. built by some other sampling scheme, such as quasi-Monte Carlo permutations — instead
of building one from scratch. Only the *shape* is checked (row count against
`n_base_samples * (n_factors + 1)`, and one permutation column per factor); the row
**contents** are trusted, not re-derived. A table of the right shape whose rows don't follow
`permutations` — or with `A`/`B` swapped — will pass and produce silently wrong effects.

# Arguments
- `factor_names` : Vector of factor names (`String` or `Symbol`).
- `n_samples` : Number of base samples to draw fresh from `factor_dist`.
- `factor_dist` : One distribution per factor, used to draw `n_samples` fresh base samples
  for both `A` and `B`.
- `strategy` : How to draw `A` and `B` — see [`SamplingStrategy`](@ref). Required, no
  default.
- `A`, `B` : Pre-built `X1`/`X2` sample pair (same shape), used directly instead of drawing
  new ones.
- `permutations` : Permutation matrix to use instead of generating a new one.
- `conditional_sampler` : Redraw function for dependent factors (keyword, optional) — see
  [`_conditionally_resample!`](@ref).
- `samples` (in the last form) : An already pick-freeze-shaped sample table, stored directly
  as the `samples` field below.
- `rng` : Random number generator (keyword, optional; first and second forms only) — governs
  permutation generation in both, and also `A`/`B`'s values in the first form under
  [`MonteCarloSampling`](@ref) (ignored by [`QuasiMonteCarloSampling`](@ref) there, which
  draws its own low-discrepancy points instead).

# Examples
```julia
using Distributions
using Random

function ishigami(X::Vector{Float64}; a::Float64=7.0, b::Float64=0.1)
    return (1 + b * X[3]^4) * sin(X[1]) + a * (sin(X[2]))^2
end

m = CallableModel(ishigami)

# With pre-defined samples `A` and `B`
du = Uniform(-π, π)
n_samples = 1000
n_factors = 3

A = DataFrame(rand(du, n_samples, n_factors), factor_names)
B = DataFrame(rand(du, n_samples, n_factors), factor_names)

s = PickAndFreezeSample(A, B)
Φₙ, Φ²ₙ = analyze(m, s)

# Drawn fresh, with an explicit sampling strategy
s = PickAndFreezeSample([:x1, :x2, :x3], n_samples, [du, du, du], MonteCarloSampling())
Φₙ, Φ²ₙ = analyze(m, s)

# With dependent factors: redraw resampled factors conditional on the frozen ones
s = PickAndFreezeSample(A, B; conditional_sampler=my_conditional_sampler)
Φₙ, Φ²ₙ = analyze(m, s)
```

$(FIELDS)
"""
struct PickAndFreezeSample
    "SAShE samples"
    samples::DataFrame

    "Permutation applied to generate samples."
    permutations::Matrix{Int64}

    function PickAndFreezeSample(samples::DataFrame, permutations::Matrix{Int64})
        _validate_pick_freeze_block(samples, permutations)
        return new(samples, permutations)
    end

    function PickAndFreezeSample(
        factor_names::Union{Vector{String}, Vector{Symbol}},
        n_samples::Int64,
        factor_dist::Vector{<:Distribution},
        strategy::SamplingStrategy;
        rng::AbstractRNG=default_rng(),
    )
        Ablock, Bblock = _draw_factor_blocks(strategy, factor_dist, [n_samples, n_samples]; rng=rng)
        A = DataFrame(Ablock, factor_names)
        B = DataFrame(Bblock, factor_names)

        X, p = _build_pick_freeze_block(A, B; rng=rng)
        return new(X, p)
    end

    function PickAndFreezeSample(
        A::DataFrame, B::DataFrame; conditional_sampler=nothing, rng::AbstractRNG=default_rng()
    )
        permutations = generate_permutations(size(A)...; rng=rng)
        return PickAndFreezeSample(A, B, permutations; conditional_sampler=conditional_sampler)
    end

    function PickAndFreezeSample(
        A::DataFrame, B::DataFrame, permutations::Matrix{Int64}; conditional_sampler=nothing
    )
        X, p = _build_pick_freeze_block(A, B, permutations)
        if !isnothing(conditional_sampler)
            _conditionally_resample!(X, A, B, permutations, conditional_sampler)
        end
        return new(X, p)
    end
end

"""
    DoubleMonteCarloSample(factor_names, factor_dist::Vector{<:Distribution}, m::Integer, strategy::SamplingStrategy; N_V=2000, N_O=1, N_I=3, rng=default_rng())
    DoubleMonteCarloSample(samples::DataFrame, permutations::Matrix{Int64}, N_V::Integer, N_O::Integer, N_I::Integer)

Pre-built sample table for the double Monte Carlo estimator ([2], §4.1, Algorithm 1),
independent factors only. Unlike [`PickAndFreezeSample`](@ref)'s row layout, which reuses
paired samples, this draws genuinely new inner/outer samples for every coalition along
every permutation — `factor_dist` must be able to produce as many fresh draws as needed,
which is why this takes distributions rather than fixed sample tables.

Row layout: the first `N_V` rows are an independent sample used only to estimate
`Var[Y]` (the cost of the full coalition). The remaining rows are grouped by permutation,
then by step along that permutation, then by outer sample — `N_O` groups of `N_I`
consecutive rows each, one group of rows per non-trivial coalition.

The second constructor wraps an already-built sample table and its bookkeeping directly —
e.g. reconstructed from an external pipeline — instead of generating one from distributions.
`samples` and `permutations` are checked for a consistent factor count and row count (the
row layout above); the values inside `samples` are still trusted, not re-derived.

# Arguments
- `factor_names` : Vector of factor names (typically String or Symbol).
- `factor_dist` : One distribution per factor — must support `rand(rng, dist)` and
  `rand(rng, dist, n)`.
- `m` : Number of random permutations to average over.
- `strategy` : How to draw every fresh row — plain Monte Carlo or a low-discrepancy
  sequence, see [`SamplingStrategy`](@ref). Required — there is no default.
- `N_V` : Sample size for the `Var[Y]` estimate — either the target when drawing fresh (first
  constructor, keyword, optional) or the value already used in `samples` (second
  constructor, positional).
- `N_O` : Number of outer samples per coalition — same two roles as `N_V` above.
- `N_I` : Number of inner samples per outer sample — must be ≥ 2 to form a sample
  variance. Unlike the nearest-neighbour estimator's structurally-fixed `N_I=1` (see
  [`DataModel`](@ref)'s estimator), this is a genuine accuracy/cost knob; [2] finds
  `N_I=3` near-optimal for a fixed total budget. Same two roles as `N_V` above.
- `rng` : Random number generator (keyword, optional; first constructor only) — governs
  permutation generation regardless of `strategy`, and also every drawn value under
  [`MonteCarloSampling`](@ref) (ignored for draws under [`QuasiMonteCarloSampling`](@ref),
  which draws its own low-discrepancy points instead).
- `samples` : Pre-built sample table (the row layout above), one row per evaluation
  (second constructor only).
- `permutations` : The permutation matrix used to build `samples` (second constructor only).

See the [References](@ref) page for the full citation behind [2].
"""
struct DoubleMonteCarloSample
    samples::DataFrame
    permutations::Matrix{Int64}
    N_V::Int64
    N_O::Int64
    N_I::Int64

    function DoubleMonteCarloSample(
        samples::DataFrame, permutations::Matrix{Int64},
        N_V::Integer, N_O::Integer, N_I::Integer,
    )
        _validate_double_monte_carlo_block(samples, permutations, N_V, N_O, N_I)
        return new(samples, permutations, N_V, N_O, N_I)
    end

    function DoubleMonteCarloSample(
        factor_names::Union{Vector{String}, Vector{Symbol}},
        factor_dist::Vector{<:Distribution},
        m::Integer,
        strategy::SamplingStrategy;
        N_V::Integer=2000, N_O::Integer=1, N_I::Integer=3, rng::AbstractRNG=default_rng(),
    )
        n_factors = length(factor_names)
        length(factor_dist) == n_factors ||
            throw(ArgumentError("factor_dist must have one distribution per factor"))
        N_I >= 2 ||
            throw(ArgumentError("N_I must be ≥ 2 to estimate a sample variance, got $N_I"))

        permutations = generate_permutations(m, n_factors; rng=rng)

        rows_per_step = N_O * N_I
        rows_per_permutation = (n_factors - 1) * rows_per_step
        total_rows = N_V + m * rows_per_permutation

        Z = zeros(total_rows, n_factors)
        Z[1:N_V, :] .= only(_draw_factor_blocks(strategy, factor_dist, [N_V]; rng=rng))
        _fill_double_monte_carlo_rows!(strategy, Z, N_V, N_O, N_I, permutations, factor_dist; rng=rng)

        return new(DataFrame(Z, factor_names), permutations, N_V, N_O, N_I)
    end
end

function _validate_pick_freeze_block(samples::DataFrame, permutations::Matrix{Int64})
    n_factors = size(samples, 2)
    n_base_samples = size(permutations, 1)
    errors::Vector{String} = []
    size(permutations, 2) == n_factors ||
        push!(errors, "`permutations` must have one column per factor in `samples`")
    size(samples, 1) == n_base_samples * (n_factors + 1) ||
        push!(errors, "`samples` must have `n_base_samples * (n_factors + 1)` rows")
    return !isempty(errors) ? error(join(errors, "\n")) : nothing
end

function _validate_double_monte_carlo_block(
    samples::DataFrame, permutations::Matrix{Int64}, N_V::Integer, N_O::Integer, N_I::Integer
)
    n_factors = size(samples, 2)
    m = size(permutations, 1)
    errors::Vector{String} = []
    size(permutations, 2) == n_factors ||
        push!(errors, "`permutations` must have one column per factor in `samples`")
    size(samples, 1) == N_V + m * (n_factors - 1) * N_O * N_I ||
        push!(errors, "`samples` must have `N_V + m * (n_factors - 1) * N_O * N_I` rows")
    return isempty(errors) ? nothing : throw(ArgumentError(join(errors, "\n")))
end

function _validate_sample_pair(X1::DataFrame, X2::DataFrame)
    size_error_msg = "`X1` and `X2` must have the same size"
    factor_names_error_msg = "`X1` and `X2` must have the same factors"
    errors::Vector{String} = []
    (size(X1) == size(X2)) || push!(errors, size_error_msg)
    names(X1) == names(X2) || push!(errors, factor_names_error_msg)
    return !isempty(errors) ? error(join(errors, "\n")) : nothing
end

"""
    _build_pick_freeze_block(A::DataFrame, B::DataFrame; rng=default_rng())
    _build_pick_freeze_block(A::DataFrame, B::DataFrame, permutations::Matrix{Int64})

Interleave an `A`/`B` sample pair into the pick-and-freeze block structure `analyze` assumes
— one untouched `A` row per base sample, followed by `d` rows progressively swapped to `B`
following a permutation (see [`_validate_pick_freeze_block`](@ref) for the shape this
produces, and the "What is `Z`?" section of [How it works](@ref) for the row layout).

Supports block generation with and without a pre-determined permutation matrix.

# Arguments
- `A`: First sample set to use
- `B`: Second sample set to use
- `permutations`: Permutations to use for cross-sampling between `A` and `B` (optional)
- `rng` : Random number generator used to generate a fresh `permutations` when none is
  given (keyword, optional; first form only).

# Returns
Tuple: Samples to be evaluated and applied permutation matrix.
"""
function _build_pick_freeze_block(A::DataFrame, B::DataFrame, permutations::Matrix{Int64})
    _validate_sample_pair(A, B)
    n_samples, n_factors = size(A)

    # Initialize generated sample
    Z = zeros(n_samples * (n_factors + 1), n_factors)

    sample_count = 1
    for i ∈ 1:n_samples
        # Leave one sample as is to create N(d+1) samples
        # where N is the number of base samples, and d is the number of factors (dimensions)
        Z[sample_count, :] .= collect(A[i, :])
        sample_count += 1

        _A = collect(A[i, :])
        _B = collect(B[i, :])
        πₙ = permutations[i, :]

        for param_idx ∈ 1:n_factors
            Zₙ = @view Z[sample_count, :]

            # Target param index is different from param_idx.
            # t_param_idx = πₙ[param_idx]
            Zₙ[πₙ[1:(param_idx)]] .= @view _B[πₙ[1:(param_idx)]]
            Zₙ[πₙ[(param_idx + 1):end]] .= @view _A[πₙ[(param_idx + 1):end]]
            sample_count += 1
        end
    end

    return DataFrame(Z, names(A)), permutations
end
function _build_pick_freeze_block(A::DataFrame, B::DataFrame; rng::AbstractRNG=default_rng())
    n_samples, n_factors = size(A)
    _permutations = generate_permutations(n_samples, n_factors; rng=rng)

    return _build_pick_freeze_block(A, B, _permutations)
end

"""
    generate_permutations(n_samples, n_factors; rng=default_rng())::Matrix{Int64}

Draw an `n_samples × n_factors` matrix of independent, uniformly random factor orderings —
row `i` is a random permutation of `1:n_factors`, the `π` [`PickAndFreezeSample`](@ref) and
[`DoubleMonteCarloSample`](@ref) walk to build their nested coalitions.

# Arguments
- `n_samples` : Number of permutations to draw (one per row).
- `n_factors` : Number of factors (length of each permutation).
- `rng` : Random number generator (keyword, optional).

# Returns
`n_samples × n_factors` `Matrix{Int64}`, each row a permutation of `1:n_factors`.
"""
function generate_permutations(
    n_samples::Int64, n_factors::Int64; rng::AbstractRNG=default_rng()
)::Matrix{Int64}
    return collect(hcat(sortperm.(eachrow(rand(rng, n_samples, n_factors)))...)')
end

"""
    _fill_double_monte_carlo_rows!(strategy, Z, N_V, N_O, N_I, permutations, factor_dist; rng)

Fill every row of `Z` after the first `N_V` (the outer/inner replicates for every coalition
step of every permutation) under `strategy` — see [`DoubleMonteCarloSample`](@ref) for the
row layout these indices follow.

Dispatches separately per `strategy` rather than sharing one loop: [`MonteCarloSampling`](@ref)
draws every replicate independently (no duplicate-point risk, so nothing to batch), while
[`QuasiMonteCarloSampling`](@ref) must draw one combined low-discrepancy point set per
coalition-step size, shared across *all* `m` permutations, and slice it — see
[`QuasiMonteCarloSampling`](@ref) for why calling the algorithm once per permutation would
silently correlate replicates that are supposed to be independent. This is a genuinely
different access pattern per strategy, not just a different random-number source, hence two
full dispatches rather than one loop parameterized by a draw function.

# Arguments
- `strategy` : How to draw the values — see [`SamplingStrategy`](@ref).
- `Z` : The sample matrix to fill in place, already sized `total_rows × n_factors`.
- `N_V`, `N_O`, `N_I` : See [`DoubleMonteCarloSample`](@ref).
- `permutations` : The `m × n_factors` permutation matrix already generated for this sample.
- `factor_dist` : One distribution per factor.
- `rng` : Random number generator (keyword; [`MonteCarloSampling`](@ref) only).

# Returns
`Z`, mutated in place.
"""
function _fill_double_monte_carlo_rows!(
    ::MonteCarloSampling, Z::AbstractMatrix, N_V::Integer, N_O::Integer, N_I::Integer,
    permutations::Matrix{Int64}, factor_dist::Vector{<:Distribution}; rng::AbstractRNG,
)
    n_perms, n_factors = size(permutations)

    row = N_V
    for ℓ ∈ 1:n_perms
        π = @view permutations[ℓ, :]
        for j ∈ 1:(n_factors - 1)
            P = @view π[1:j]
            notP = @view π[(j + 1):end]
            for _ ∈ 1:N_O
                outer_vals = [rand(rng, factor_dist[i]) for i ∈ notP]
                for _ ∈ 1:N_I
                    row += 1
                    for (idx, i) ∈ enumerate(notP)
                        Z[row, i] = outer_vals[idx]
                    end
                    for i ∈ P
                        Z[row, i] = rand(rng, factor_dist[i])
                    end
                end
            end
        end
    end
    return Z
end
function _fill_double_monte_carlo_rows!(
    strategy::QuasiMonteCarloSampling, Z::AbstractMatrix, N_V::Integer, N_O::Integer, N_I::Integer,
    permutations::Matrix{Int64}, factor_dist::Vector{<:Distribution}; rng::AbstractRNG,
)
    n_perms, n_factors = size(permutations)
    rows_per_step = N_O * N_I
    rows_per_permutation = (n_factors - 1) * rows_per_step

    for j ∈ 1:(n_factors - 1)
        d_notP = n_factors - j
        d_P = j

        # One combined low-discrepancy draw per (block type, j) across *all* m
        # permutations, not one call per permutation, so a deterministic algorithm
        # can't hand identical points to two permutations sharing this step index.
        # The raw [0,1] points are shared; which factor each column maps to is
        # resolved per permutation below via `quantile`.
        outer_unit = permutedims(QMC.sample(n_perms * N_O, d_notP, strategy.algorithm))
        inner_unit = permutedims(QMC.sample(n_perms * N_O * N_I, d_P, strategy.algorithm))

        outer_row = 0
        inner_row = 0
        for ℓ ∈ 1:n_perms
            π = @view permutations[ℓ, :]
            P = @view π[1:j]
            notP = @view π[(j + 1):end]
            base = N_V + (ℓ - 1) * rows_per_permutation + (j - 1) * rows_per_step

            for l ∈ 1:N_O
                outer_row += 1
                outer_vals = [
                    quantile(factor_dist[i], outer_unit[outer_row, idx]) for
                    (idx, i) ∈ enumerate(notP)
                ]
                for h ∈ 1:N_I
                    inner_row += 1
                    row = base + (l - 1) * N_I + h
                    for (idx, i) ∈ enumerate(notP)
                        Z[row, i] = outer_vals[idx]
                    end
                    for (idx, i) ∈ enumerate(P)
                        Z[row, i] = quantile(factor_dist[i], inner_unit[inner_row, idx])
                    end
                end
            end
        end
    end
    return Z
end

"""
    _conditionally_resample!(Z, A, B, permutations, conditional_sampler)

Overwrite the resampled-factor columns of every non-base row of a generated sample `Z`
(as returned by [`_build_pick_freeze_block`](@ref)) with draws from `conditional_sampler`.

Needed when some factors are statistically dependent: the pick-freeze construction in
`_build_pick_freeze_block` assumes `(xᵤ, y₋ᵤ)` is a valid draw from the joint distribution, which
only holds under independence.

`conditional_sampler(X1_param_idx, X2_param_idx, base_row, noise_row)` must return the
values for the factors in `X2_param_idx`, in that order, drawn conditional on the factors
in `X1_param_idx` being fixed at their `base_row` (i.e. `A`) values.

!!! warning "`noise_row` is a draw from the joint input distribution, not independent noise"
    `noise_row` is `B[i, :]` — a row of the second sample — and all `d` calls for base sample
    `i` receive that same row. Its coordinates therefore carry the factors' own dependence
    structure. Using it to *generate* a conditional draw is fine as long as you transform it
    correctly for the distribution you need (e.g. affinely remapping a uniform coordinate onto
    a conditional uniform support, as `test/conditional.jl` does). Feeding it into a formula
    that expects *independent* noise is not: substituting it for independent standard normals
    in a Gaussian conditional formula measured ≈24% high on two of three factors (≈44 standard
    errors) on a correlated-Gaussian model.

    Reusing the same row across all `d` positions does **not** bias the point estimate
    (measured unbiased, `|z| ≤ 1.9`, on a constrained `x₃ = x₁ + x₂` model with exact effects
    known). [4] §5.2 does ask for draws that are *"independent of the existing ones"* across
    positions; the effect of not having that appears in interval width rather than in the
    estimate, and is not measured here.

The per-row resamples are dispatched with `pmap`.
"""
function _conditionally_resample!(
    Z::DataFrame,
    A::DataFrame,
    B::DataFrame,
    permutations::Matrix{Int64},
    conditional_sampler,
)
    n_samples, n_factors = size(A)
    block_size = n_factors + 1

    tasks = map(1:(n_samples * n_factors)) do k
        i = div(k - 1, n_factors) + 1        # base sample
        pos = mod(k - 1, n_factors) + 1      # position within the pick-freeze block
        πₙ = permutations[i, :]
        target_row = (i - 1) * block_size + 1 + pos
        (target_row, πₙ[1:pos], πₙ[(pos + 1):end], collect(A[i, :]), collect(B[i, :]))
    end

    results = pmap(tasks) do (target_row, X2_param_idx, X1_param_idx, base_row, noise_row)
        values = conditional_sampler(X1_param_idx, X2_param_idx, base_row, noise_row)
        (target_row, X2_param_idx, values)
    end

    for (target_row, X2_param_idx, values) ∈ results
        for (col, value) ∈ zip(X2_param_idx, values)
            Z[target_row, col] = value
        end
    end

    return Z
end

