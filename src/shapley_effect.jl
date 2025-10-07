
"""
    problem (X1::DataFrame, X2::DataFrame, Y::Vector, Y⁻::Matrix, Y⁺::Matrix)
    problem (X1::DataFrame, X2::DataFrame)

# Arguments
- `func` : A function that accepts a vector of inputs as argument
- `X1` : First set of samples (of the same size as X2) to be used as inputs to func
- `X2` : Second set of samples (of the same size as X1) to be used as inputs to func
- `Y:` : Value of the model calculated for each sample row of X1 (initialized empty)
- `Y⁻` : Matrix of intermediate model values for each pair sample/factor (initialized empty)
- `Y⁺` : Matrix of intermediate model values for each pair sample/factor (initialized empty)
- `Φ_increments` : Contribution that each sample iteration (rows) gives to each factor
(cols) total ShapleyEffect
- `Φ²_increments` : Contribution that each sample iteration (rows) gives to each Shapley
Effect squared expected valued

# References
1. Goda, T., 2021. A simple algorithm for global sensitivity analysis with Shapley effects. \
   Reliability Engineering & System Safety 213, 107702. \
   https://doi.org/10.1016/j.ress.2021.107702
"""
struct Problem
    func::Function
    X1::DataFrame                       # What the paper calls x
    X2::DataFrame                       # What the paper calls y
    Y::Vector                           # What the paper calls F
    Y⁻::Matrix                          # What the paper calls F⁻
    Y⁺::Matrix                          # What the paper calls F⁺
    permutations::Matrix                # What the paper calls π
    Φ_increments::Matrix{Float64}
    Φ²_increments::Matrix{Float64}
    n_samples::Int64

    # TODO Rename Problem to Model?
    function Problem(func::Function, X1::DataFrame, X2::DataFrame)
        # Validate inputs
        _validate_problem(X1, X2)

        n_samples, n_factors = size(X1)
        _Y::Vector = zeros(Float64, n_samples)
        # TODO Switch rows and cols so each col is a sample to improve performance
        _Y⁻::Matrix = zeros(Float64, n_samples, n_factors)
        _Y⁺::Matrix = zeros(Float64, n_samples, n_factors)
        _permutations = collect(hcat(sortperm.(eachrow(rand(n_samples, n_factors)))...)')
        _Φ_increments::Matrix{Float64} = zeros(Float64, n_samples, n_factors)
        _Φ²_increments::Matrix{Float64} = zeros(Float64, n_samples, n_factors)

        return new(
            func,
            X1,
            X2,
            _Y,
            _Y⁻,
            _Y⁺,
            _permutations,
            _Φ_increments,
            _Φ²_increments,
            n_samples
        )
    end
end

function _validate_problem(X1::DataFrame, X2::DataFrame)
    size_error_msg = "`samples_X1` and `samples_X2` must have the same size"
    factor_names_error_msg = "`samples_X1` and `samples_X2` must have the same factors"
    errors::Vector{String} = []
    (size(X1) == size(X2)) || push!(errors, size_error_msg)
    names(X1) == names(X2) || push!(errors, factor_names_error_msg)
    return !isempty(errors) ? error(join(errors, "\n")) : nothing
end

function margin_of_error(Φₙ::Matrix{Float64}, Φ²ₙ::Matrix{Float64})::Vector{Float64}
    n_samples = size(Φₙ, 2)
    # E[Φ²] == sum(Φ²ₙ, dims=2) and (E[Φ])² == (sum(Φₙ, dims=2)).^2
    stds = sqrt.(
        (shapley_effects(Φ²ₙ) .- (shapley_effects(Φₙ) .^ 2)) ./ (n_samples - 1)
    )
    return 1.96 .* stds
end

function confint(Φₙ::Matrix{Float64}, Φ²ₙ::Matrix{Float64})
    Φ = shapley_effects(Φₙ)
    moe = margin_of_error(Φₙ, Φ²ₙ)
    Φ .- moe, Φ .+ moe
end

"""
    shapley_effects(Φₙ)::Vector{Float64}

Estimate Shapley effects for each factor.

# Arguments
- `Φₙ` : Shapley effects for each sample (rows) and factor (cols)

# Returns
Vector of size `d` (dimensionality of the problem), containing the estimated Shapley
effects.
"""
function shapley_effects(Φₙ)::Vector{Float64}
    return dropdims(sum(Φₙ, dims=2), dims=2)
end

"""
    shapley_effects(Φₙ, Φ²ₙ)::NTuple{3,Vector{Float64}}

# Arguments
- `Φₙ` : Shapley effects for each sample (rows) and factor (cols)
- `Φ²ₙ` : Variance of the Shapley effects for each sample (rows) and factor (cols)

# Returns
Tuple, of vectors with same dimensionality of the problem.
- Φ (Shapley effect)
- Φ + stdev
- Φ - stdev
"""
function shapley_effects(Φₙ, Φ²ₙ)::NTuple{3,Vector{Float64}}
    Φ = shapley_effects(Φₙ)
    moe = margin_of_error(Φₙ, Φ²ₙ)
    Φ, Φ .- moe, Φ .+ moe
end

"""
    _shapley_effect_dependent_iteration(
        func::Function,
        iter_idx::Int64,
        X1ₙ::DataFrameRow,
        X2ₙ::DataFrameRow,
        sample_X2::Function,
        πₙ::AbstractVector{Int64},          # A row of the permutation matrix.
        Yₙ⁻::AbstractVector{Float64},
        Yₙ⁺::AbstractVector{Float64},
        Φₙ_increments::AbstractVector{Float64},
        Φₙ²_increments::AbstractVector{Float64},
        n_samples::Int64,
    )

Allows the user to specify a function `sample_X2` that samples X2 conditionally with respect
to X1 at each iteration. That's necessary when some factors are dependent.

#TODO Give more details.

"""
function _shapley_effect_dependent_iteration(
    func::Function,
    sample_X2::Function,
    X1ₙ::DataFrameRow,
    X2ₙ::DataFrameRow,
    πₙ::AbstractVector{Int64},          # A row of the permutation matrix.
    Yₙ⁻::AbstractVector{Float64},
    Yₙ⁺::AbstractVector{Float64},
    Φₙ_increments::AbstractVector{Float64},
    Φₙ²_increments::AbstractVector{Float64},
    n_samples::Int64,
)
    _X1 = collect(X1ₙ)
    _X2 = collect(X2ₙ)
    factor_names = names(X1ₙ)
    Zₙ = copy(_X1)
    Yₙ = func(Zₙ)
    Yₙ⁻[πₙ[1]] = Yₙ

    n_var_params = length(X1ₙ)
    t_param_idx::Int64 = 1
    for param_idx ∈ 1:n_var_params
        # Target param index is different from param_idx.
        t_param_idx = πₙ[param_idx]

        X1_param_idx = πₙ[(param_idx+1):end]
        X2_param_idx = πₙ[1:(param_idx)]

        # sample_X2 is expected to return a Vector with sampled values for factors X2_param_idx
        Zₙ[X2_param_idx] .= sample_X2(X1_param_idx, X2_param_idx, _X1, _X2, factor_names)
        Zₙ[X1_param_idx] .= @view _X1[X1_param_idx]

        Yₙ⁺[t_param_idx] = func(Zₙ)

        f_diff = (Yₙ⁻[t_param_idx] - Yₙ⁺[t_param_idx])
        f_arg = (Yₙ - (Yₙ⁻[t_param_idx] / 2) - (Yₙ⁺[t_param_idx] / 2)) * f_diff

        Φₙ_increments[t_param_idx] = f_arg * (1 / n_samples)
        Φₙ²_increments[t_param_idx] = f_arg^2 * (1 / n_samples)

        if param_idx < n_var_params
            Yₙ⁻[πₙ[param_idx+1]] = Yₙ⁺[t_param_idx]
        end
    end
    if any(isnan.(Φₙ_increments))
        error("NaN detected in Shapley effect increments")
    end

    # TODO Maybe we could return a solution object with the below plus Φ and Φ²
    return (Φₙ_increments, Φₙ²_increments, Yₙ)
end

function _shapley_effect_iteration(
    func::Function,
    X1ₙ::DataFrameRow,
    X2ₙ::DataFrameRow,
    πₙ::AbstractVector{Int64},          # A row of the permutation matrix.
    Yₙ⁻::AbstractVector{Float64},
    Yₙ⁺::AbstractVector{Float64},
    Φₙ_increments::AbstractVector{Float64},
    Φₙ²_increments::AbstractVector{Float64},
    n_samples::Int64,
)
    _X1 = collect(X1ₙ)
    _X2 = collect(X2ₙ)
    Zₙ = collect(X1ₙ)
    Yₙ = func(_X1)
    Yₙ⁻[πₙ[1]] = Yₙ

    n_var_params = length(X1ₙ)
    t_param_idx::Int64 = 1
    for param_idx ∈ 1:n_var_params
        # Target param index is different from param_idx.
        t_param_idx = πₙ[param_idx]

        Zₙ[πₙ[1:(param_idx)]] .= @view _X2[πₙ[1:(param_idx)]]
        Zₙ[πₙ[(param_idx+1):end]] .= @view _X1[πₙ[(param_idx+1):end]]

        Yₙ⁺[t_param_idx] = func(Zₙ)

        f_diff = (Yₙ⁻[t_param_idx] - Yₙ⁺[t_param_idx])
        f_arg = (Yₙ - Yₙ⁻[t_param_idx] / 2 - Yₙ⁺[t_param_idx] / 2) * f_diff

        Φₙ_increments[t_param_idx] = f_arg * (1 / n_samples)
        Φₙ²_increments[t_param_idx] = f_arg^2 * (1 / n_samples)

        if param_idx < n_var_params
            Yₙ⁻[πₙ[param_idx+1]] = Yₙ⁺[t_param_idx]
        end
    end

    # TODO Maybe we could return a solution object with the below plus Φ and Φ²
    return (Φₙ_increments, Φₙ²_increments, Yₙ)
end

"""
    solve(problem::Problem; dependent::Bool=false, sample_X2::Function=nothing)

Evaluate `Problem` with associated model and determine Shapley effect.

TODO Rename `solve` to `shapley_effect` or `analyze`?

# Arguments
- `problem` : SAShE Problem
- `dependent` : Defaults to false. When true, use alternative algorithm for dependent factors.
- `sample_X2` : To be used when `dependent == true`.  More details can be found in
`_shapley_effect_dependent_iteration`'s docstring.

# Returns
Tuple of matrices: Φₙ, Φ²ₙ, Yₙ
- Φₙ : Shapley effects for base samples (size `N`)
- Φ²ₙ : Variance of Shapley effects used to estimate confidence bounds
- Yₙ : Corresponding model result for base samples (size `N`)
"""
function solve(problem::Problem)
    n_samples = problem.n_samples

    res = if dependent
        @showprogress pmap(
            _shapley_effect_dependent_iteration,
            repeated(problem.func, n_samples),
            repeated(sample_X2, n_samples),
            eachrow(problem.X1),
            eachrow(problem.X2),
            eachrow(problem.permutations),
            eachrow(problem.Y⁻),
            eachrow(problem.Y⁺),
            eachrow(problem.Φ_increments),
            eachrow(problem.Φ²_increments),
            repeated(problem.n_samples, n_samples)
        )
    else
        @showprogress pmap(
            _shapley_effect_iteration,
            repeated(problem.func, n_samples),
            eachrow(problem.X1),
            eachrow(problem.X2),
            eachrow(problem.permutations),
            eachrow(problem.Y⁻),
            eachrow(problem.Y⁺),
            eachrow(problem.Φ_increments),
            eachrow(problem.Φ²_increments),
            repeated(problem.n_samples, n_samples)
        )
    end

    # TODO Return a better object, either a `Solution` or a new version of `Problem`
    return hcat([r[1] for r in res]...), hcat([r[2] for r in res]...), [r[3] for r in res]
end

"""
    analyze(X::DataFrame, Y::Vector, perms::Matrix)

# Arguments
- `X` : Inputs used to run target model
- `Y` : Resulting outputs from `X`
- `perms` : Permutation order

# Returns
Tuple, of Φₙ and Φₙ² (Shapley Effect and variance)
"""
function analyze(X::DataFrame, Y::Vector, perms::Matrix)
    n_var_params = size(X, 2)
    n_base_samples = size(perms, 1)

    check_size = Int64(size(X, 1) / (n_var_params + 1))
    @assert check_size == n_base_samples "Sample sizes do not match!"

    Φₙ_increments = zeros(n_base_samples, n_var_params)
    Φₙ²_increments = zeros(n_base_samples, n_var_params)

    Yₙ⁻ = zeros(n_var_params)
    Yₙ⁺ = zeros(n_var_params)

    for n in 1:n_base_samples
        # For each sample...

        # Calculate the starting index for this base sample
        base_idx = (n - 1) * (n_var_params + 1) + 1

        # Yₙ⁻ is the result for Xₙ, and gets replaced by Yₙ⁺
        # Yₙ⁺ is the result for Xₙ₊₁
        πₙ = perms[n, :]
        Yₙ = Y[base_idx]

        Yₙ⁻ .= 0.0
        Yₙ⁺ .= 0.0
        Yₙ⁻[πₙ[1]] = Yₙ  # Set first value according to permutation

        for param_idx in 1:n_var_params
            eval_idx = base_idx + param_idx
            t_param_idx = πₙ[param_idx]

            Yₙ⁺[t_param_idx] = Y[eval_idx]

            f_diff = (Yₙ⁻[t_param_idx] - Yₙ⁺[t_param_idx])
            f_arg = (Yₙ - Yₙ⁻[t_param_idx] / 2 - Yₙ⁺[t_param_idx] / 2) * f_diff

            Φₙ_increments[n, t_param_idx] = f_arg * (1 / n_base_samples)
            Φₙ²_increments[n, t_param_idx] = f_arg^2 * (1 / n_base_samples)

            if param_idx < n_var_params
                Yₙ⁻[πₙ[param_idx+1]] = Yₙ⁺[t_param_idx]
            end
        end
    end

    return (Matrix(Φₙ_increments'), Matrix(Φₙ²_increments'))
end

"""
    analyze(S::SAShESample, Y::Vector)

# Arguments
- `S` : SAShE sample
- `Y` : Model results

# Returns
Tuple, of Φₙ and Φₙ² (Shapley Effect and variance)
"""
function analyze(S::SAShESample, Y::Vector)
    X = S.samples
    return analyze(X, Y, S.permutations)
end

function Base.:show(io::IO, p::Problem)
    println(p.func)
    println("n_samples: ", p.n_samples)
end
