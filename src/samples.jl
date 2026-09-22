using DataFrames
using Random
using Distributions
import QuasiMonteCarlo as QMC

"""
    _generate_samples(A::DataFrame, B::DataFrame)
    _generate_samples(A::DataFrame, B::DataFrame, permutations::Matrix{Int64})
    _generate_samples(A::DataFrame, B::DataFrame, sampler)

Generate samples for use with SAShE.

Supports sample generation with and without a pre-determined permutation matrix.

# Arguments
- `A`: First sample set to use
- `B`: Second sample set to use
- `permutations`: Permutations to use for cross-sampling between `A` and `B` (optional)
- `sampler` : Method to generate permutations (optional)

# Returns
Tuple: Samples to be evaluated and applied permutation matrix.
"""
function _generate_samples(A::DataFrame, B::DataFrame, permutations::Matrix{Int64})
    SAShE._validate_callable_model(A, B)
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
function _generate_samples(A::DataFrame, B::DataFrame)
    n_samples, n_factors = size(A)
    _permutations = generate_permutations(n_samples, n_factors)

    return _generate_samples(A, B, _permutations)
end
function _generate_samples(A::DataFrame, B::DataFrame, sampler)
    n_samples, n_factors = size(A)
    permutations = collect(
        hcat(sortperm.(eachcol(QMC.sample(n_samples, n_factors, sampler())))...)'
    )

    return _generate_samples(A, B, permutations)
end

function generate_permutations(n_samples::Int64, n_factors::Int64)::Matrix{Int64}
    return collect(hcat(sortperm.(eachrow(rand(n_samples, n_factors)))...)')
end

"""
    _conditionally_resample!(Z, A, B, permutations, conditional_sampler)

Overwrite the resampled-factor columns of every non-base row of a generated sample `Z`
(as returned by [`_generate_samples`](@ref)) with draws from `conditional_sampler`.

Needed when some factors are statistically dependent: the pick-freeze construction in
`_generate_samples` assumes `(xᵤ, y₋ᵤ)` is a valid draw from the joint distribution, which
only holds under independence.

`conditional_sampler(X1_param_idx, X2_param_idx, base_row, noise_row)` must return the
values for the factors in `X2_param_idx`, in that order, drawn conditional on the factors
in `X1_param_idx` being fixed at their `base_row` (i.e. `A`) values. `noise_row` (from `B`)
carries an independent draw usable as a randomness source.

The per-row resamples are independent and are dispatched with `pmap`.
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

"""
    _even_split(factor_names, X)

Split a given sample set `X` into two equally sized DataFrames.
"""
function _even_split(factor_names, X)
    n_factors = length(factor_names)
    n_samples = size(X, 1)
    half_size = Int64(n_samples / 2)
    A = DataFrame(zeros(half_size, n_factors), factor_names)
    B = copy(A)

    A[:, :] = X[1:half_size, :]
    B[:, :] = X[(half_size + 1):end, :]

    return A, B
end

"""
    create_sample(factor_names::Vector, n_samples::Int64, factor_dist::Vector)
    create_sample(factor_names::Vector, X::Matrix)

Create samples for use with SAShE.

Supports random sampling by creating new samples:
1. Based on known distributions
2. From an existing base sample `X` \
   Sample `X` must be divisible by 2.

The generated sample size will be `N(d+1)`, where `N` is the base sample size and `d` is the
number of factors (i.e., the dimensionality of the problem).

# Arguments
- `factor_names`: Vector of factor names (typically String or Symbol)
- `n_samples`: Number of base samples desired.
- `factor_dist`: Distributions for each factor
- `X`: Base sample to use

# Returns
Tuple: Samples to be evaluated and applied permutation matrix.
"""
function create_sample(factor_names::Vector, n_samples::Int64, factor_dist::Vector)
    A, B = _even_split(factor_names, zeros(n_samples, n_factors))

    for (i, fd) ∈ enumerate(factor_dist)
        A[:, i] .= rand(fd, n_samples)
        B[:, i] .= rand(fd, n_samples)
    end

    return _generate_samples(A, B)
end
function create_sample(factor_names::Vector, X::Matrix)
    A, B = _even_split(factor_names, X)

    return _generate_samples(A, B)
end

"""
    CallableModelSample(factor_names::Union{Vector{String},Vector{Symbol}}, n_samples::Int64, factor_dist::Vector)
    CallableModelSample(factor_names::Union{Vector{String},Vector{Symbol}}, samples::Matrix)
    CallableModelSample(factor_names::Union{Vector{String},Vector{Symbol}}, samples::Matrix, sampler)
    CallableModelSample(A::DataFrame, B::DataFrame; conditional_sampler=nothing)
    CallableModelSample(A::DataFrame, B::DataFrame, permutations::Matrix{Int64}; conditional_sampler=nothing)

SAShE samples (`X`) and permutations (`π`).

Pass `conditional_sampler` when some factors are dependent: after the pick-freeze samples
are built, every non-base row has its resampled factors redrawn conditional on the frozen
ones. See [`_conditionally_resample!`](@ref) for the expected signature.

# Examples
```julia
import QuasiMonteCarlo as QMC
using Distributions
using Random

function ishigami(X::Vector{Float64}; a::Float64=7.0, b::Float64=0.1)
    return (1 + b * X[3]^4) * sin(X[1]) + a * (sin(X[2]))^2
end

# With pre-defined samples `A` and `B`
du = Uniform(-π, π)
n_samples = 1000
n_factors = 3

# Samples with shape [n_samples ⋅ n_factors] and known permutations
A = DataFrame(rand(du, n_samples, n_factors), factor_names)
B = DataFrame(rand(du, n_samples, n_factors), factor_names)
permutations = ...

S_x = CallableModelSample(A, B, permutations)
Y = map(x -> ishigami(collect(x)), eachrow(S_x.samples))
Φₙ, Φ²ₙ = analyze(S_x, Y)

# With pre-defined samples (halved and split into `A` and `B`)
X = Matrix(QMC.sample(2048, fill(-π, 3), fill(Float64(π), 3), Uniform())')
S_x = CallableModelSample([:x1, :x2, :x3], X)
Y = map(x -> ishigami(collect(x)), eachrow(S_x.samples))
Φₙ, Φ²ₙ = analyze(S_x, Y)

# With a given sampler to create custom permutations
X = Matrix(QMC.sample(2048, fill(-π, 3), fill(Float64(π), 3), Uniform())')
S_x = CallableModelSample([:x1, :x2, :x3], X, QMC.LatinHypercubeSample())
Y = map(x -> ishigami(collect(x)), eachrow(S_x.samples))
Φₙ, Φ²ₙ = analyze(S_x, Y)

# With dependent factors: redraw resampled factors conditional on the frozen ones
S_x = CallableModelSample(A, B; conditional_sampler=my_conditional_sampler)
Y = map(x -> ishigami(collect(x)), eachrow(S_x.samples))
Φₙ, Φ²ₙ = analyze(S_x, Y)
```

$(FIELDS)
"""
struct CallableModelSample
    "SAShE samples"
    samples

    "Permutation applied to generate samples."
    permutations

    function CallableModelSample(problem::CallableModel)
        X, p = _generate_samples(problem.X1, problem.X2, problem.permutations)
        return new(X, p)
    end
    function CallableModelSample(
        factor_names::Union{Vector{String}, Vector{Symbol}},
        n_samples::Int64,
        factor_dist::Vector,
    )
        X, p = create_sample(factor_names, n_samples, factor_dist)
        return new(X, p)
    end

    function CallableModelSample(
        factor_names::Union{Vector{String}, Vector{Symbol}}, samples::Matrix
    )
        X, p = create_sample(factor_names, samples)
        return new(X, p)
    end

    function CallableModelSample(
        factor_names::Union{Vector{String}, Vector{Symbol}}, samples::Matrix, sampler
    )
        A, B = _even_split(factor_names, samples)
        X, p = _generate_samples(A, B, sampler)
        return new(X, p)
    end

    function CallableModelSample(A::DataFrame, B::DataFrame; conditional_sampler=nothing)
        permutations = generate_permutations(size(A)...)
        return CallableModelSample(A, B, permutations; conditional_sampler=conditional_sampler)
    end

    function CallableModelSample(
        A::DataFrame, B::DataFrame, permutations::Matrix{Int64}; conditional_sampler=nothing
    )
        X, p = _generate_samples(A, B, permutations)
        if !isnothing(conditional_sampler)
            _conditionally_resample!(X, A, B, permutations, conditional_sampler)
        end
        return new(X, p)
    end
end
