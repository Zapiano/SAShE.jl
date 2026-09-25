using Random: AbstractRNG, default_rng
using Distributions: Distribution, quantile
import QuasiMonteCarlo as QMC

"""
    SamplingStrategy

Abstract supertype for how [`CallablePickAndFreezeSample`](@ref) and [`CallableDoubleMonteCarloSample`](@ref)
draw fresh factor values from `factor_dist`. Concrete subtypes select the draw mechanism
independently of which sample type uses them — the same role [`EstimationMethod`](@ref)
plays for choosing an estimator.
"""
abstract type SamplingStrategy end

"""
    MonteCarloSampling()

Draw factor values by ordinary i.i.d. Monte Carlo (`rand`) — the draw mechanism the
pick-and-freeze and double Monte Carlo convergence results ([1], [2], [4]) are stated for.

See the [References](@ref) page for the full citations behind [1], [2], and [4].
"""
struct MonteCarloSampling <: SamplingStrategy end

"""
    QuasiMonteCarloSampling(algorithm=QuasiMonteCarlo.LatinHypercubeSample())

Draw factor values from a stratified/low-discrepancy point set instead of i.i.d. Monte Carlo,
using `algorithm` from QuasiMonteCarlo.jl. Each factor's own marginal distribution is respected
via the probability integral transform: `algorithm`'s point set on `[0,1]^d` is pushed through
`quantile(dist, ·)` per factor — not through QuasiMonteCarlo.jl's own
`Distributions.Sampleable` support, which is plain `rand` under the hood, not low-discrepancy.

This is an **optional extension**: none of the papers behind this package use it, and it has
no measured accuracy advantage over [`MonteCarloSampling`](@ref) here (see
[Deliberate deviations](@ref)). It is available for experimentation; `MonteCarloSampling()` is
what the documented workflows use.

Every draw that must be mutually independent (each coalition step's outer/inner replicates in
`CallableDoubleMonteCarloSample`) is generated as **one** combined low-discrepancy point set, sliced
into disjoint blocks, rather than calling `algorithm` once per block.

!!! warning "Only randomized algorithms are accepted"
    `algorithm` must be a `QuasiMonteCarlo.RandomSamplingAlgorithm` (e.g.
    `LatinHypercubeSample`), which randomizes each dimension independently. Deterministic
    low-discrepancy sequences (`SobolSample`, `HaltonSample`, …, including scrambled or
    shifted variants) are **rejected**, because they give silently wrong answers for *both*
    estimators — measured, not merely unproven:

    - **Pick-and-freeze** needs `A` and `B` to be independent, because it builds hybrid rows
      taking some coordinates from `A` and the rest from `B`. Blocks sliced out of a *single*
      deterministic point set are not: measured at `n=4096, d=3`, `SobolSample()` gives
      `cor(A₁, B₁) = 1.0000` with `mean|A₁ − B₁| = 2.4e-4` — the two halves are effectively
      *identical* in the first coordinate (`HaltonSample()` likewise), so swapping that factor
      changes nothing and its effect collapses. On Ishigami at `n=4096` this reported factor
      1's effect as `9.2e-5` against an exact value of `6.0327` (see
      `test/reference_values.jl` for that value's derivation), recovering 2.85 of `13.8446`
      output variance.
    - **Double Monte Carlo** estimates `Var[Y|X₋ₚ]` from the sample variance of `N_I` inner
      replicates. Under a low-discrepancy sequence those replicates are stratified by
      construction, so their sample variance is a biased estimator of that conditional
      variance. On Ishigami over 8 seeds, `SobolSample()` is biased by 17% and 38% on two of
      three factors (≈10 standard errors) — a consistent offset, not noise.

**Caveat**: the convergence results for both estimators are established in [1], [2], [4] for
genuinely i.i.d. draws. Whether they still hold under a *randomized* low-discrepancy design
is not established by the cited papers — and [4]'s Remark 1 argues the improvement is hard to
obtain at all for Shapley effects, since generating the permutation breaks the smoothness QMC
theory relies on. No measured accuracy gain over [`MonteCarloSampling`](@ref) has been
demonstrated here. Treat results under this strategy as exploratory.

!!! note "Reproducibility"
    The `rng` keyword threaded through `CallablePickAndFreezeSample`/`CallableDoubleMonteCarloSample` seeds
    the permutations, but **not** this strategy's point set — the algorithm carries its own
    generator. For a reproducible run, construct the algorithm with an explicit one, e.g.
    `QuasiMonteCarloSampling(QuasiMonteCarlo.LatinHypercubeSample(; rng=Xoshiro(1)))`.

# Arguments
- `algorithm` : A `QuasiMonteCarlo.RandomSamplingAlgorithm` instance.

See the [References](@ref) page for the full citations behind [1], [2], and [4].
"""
struct QuasiMonteCarloSampling{A <: QMC.SamplingAlgorithm} <: SamplingStrategy
    algorithm::A

    function QuasiMonteCarloSampling(
        algorithm::A=QMC.LatinHypercubeSample()
    ) where {A <: QMC.SamplingAlgorithm}
        algorithm isa QMC.RandomSamplingAlgorithm || throw(
            ArgumentError(
                "$(nameof(A)) is a deterministic low-discrepancy sequence, which gives " *
                "silently wrong Shapley effects for both estimators here. Blocks sliced " *
                "from one such point set are not independent (measured: the two halves are " *
                "identical in the first coordinate), which breaks pick-and-freeze's hybrid " *
                "rows; and its points are stratified rather than i.i.d., which biases " *
                "double Monte Carlo's inner sample variance. Use a " *
                "QuasiMonteCarlo.RandomSamplingAlgorithm such as LatinHypercubeSample(), " *
                "which randomizes each dimension independently, or MonteCarloSampling().",
            ),
        )
        return new{A}(algorithm)
    end
end

"""
    _draw_factor_blocks(strategy::SamplingStrategy, factor_dist::Vector{<:Distribution}, block_sizes::Vector{<:Integer}; rng=default_rng())::Vector{Matrix{Float64}}

Draw `sum(block_sizes)` fresh rows of `factor_dist` under `strategy`, split into
`length(block_sizes)` disjoint blocks of the given sizes, in order — see
[`QuasiMonteCarloSampling`](@ref) for why drawing from one combined point set (rather than
calling `strategy` once per block) matters for independence.

# Arguments
- `strategy` : How to draw the values — see [`SamplingStrategy`](@ref).
- `factor_dist` : One distribution per factor, i.e. per column of every returned block.
- `block_sizes` : Row count of each block to return, in order.
- `rng` : Random number generator (keyword, optional; [`MonteCarloSampling`](@ref) only).

# Returns
`Vector` of `length(block_sizes)` matrices, block `k` sized `block_sizes[k] × length(factor_dist)`.
"""
function _draw_factor_blocks(
    ::MonteCarloSampling, factor_dist::Vector{<:Distribution}, block_sizes::Vector{<:Integer};
    rng::AbstractRNG=default_rng(),
)::Vector{Matrix{Float64}}
    return map(block_sizes) do n
        reduce(hcat, [rand(rng, fd, n) for fd ∈ factor_dist])
    end
end

function _draw_factor_blocks(
    strategy::QuasiMonteCarloSampling, factor_dist::Vector{<:Distribution},
    block_sizes::Vector{<:Integer}; rng::AbstractRNG=default_rng(),
)::Vector{Matrix{Float64}}
    n_factors = length(factor_dist)
    total = sum(block_sizes)
    U = permutedims(QMC.sample(total, n_factors, strategy.algorithm))  # total × n_factors, in [0,1]

    blocks = Matrix{Float64}[]
    start = 1
    for n ∈ block_sizes
        Ublock = @view U[start:(start + n - 1), :]
        push!(
            blocks,
            reduce(hcat, [quantile.(fd, @view Ublock[:, i]) for (i, fd) ∈ enumerate(factor_dist)]),
        )
        start += n
    end
    return blocks
end
