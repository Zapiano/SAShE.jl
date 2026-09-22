function margin_of_error(Φₙ::Matrix{Float64}, Φ²ₙ::Matrix{Float64})::Vector{Float64}
    n_samples = size(Φₙ, 2)
    # E[Φ²] == sum(Φ²ₙ, dims=2) and (E[Φ])² == (sum(Φₙ, dims=2)).^2
    stds = sqrt.((shapley_effects(Φ²ₙ) .- (shapley_effects(Φₙ) .^ 2)) ./ (n_samples - 1))
    return 1.96 .* stds
end

function confint(Φₙ::Matrix{Float64}, Φ²ₙ::Matrix{Float64})
    Φ = shapley_effects(Φₙ)
    moe = margin_of_error(Φₙ, Φ²ₙ)
    return Φ .- moe, Φ .+ moe
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
function shapley_effects(Φₙ::AbstractArray{Float64, 2})::Vector{Float64}
    return dropdims(sum(Φₙ; dims=2); dims=2)
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
function shapley_effects(
    Φₙ::AbstractArray{Float64, 2}, Φ²ₙ::AbstractArray{Float64, 2}
)::NTuple{3, Vector{Float64}}
    Φ = shapley_effects(Φₙ)
    moe = margin_of_error(Φₙ, Φ²ₙ)
    return Φ, Φ .- moe, Φ .+ moe
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
        Zₙ[πₙ[(param_idx + 1):end]] .= @view _X1[πₙ[(param_idx + 1):end]]

        Yₙ⁺[t_param_idx] = func(Zₙ)

        f_diff = (Yₙ⁻[t_param_idx] - Yₙ⁺[t_param_idx])
        f_arg = (Yₙ - Yₙ⁻[t_param_idx] / 2 - Yₙ⁺[t_param_idx] / 2) * f_diff

        Φₙ_increments[t_param_idx] = f_arg * (1 / n_samples)
        Φₙ²_increments[t_param_idx] = f_arg^2 * (1 / n_samples)

        if param_idx < n_var_params
            Yₙ⁻[πₙ[param_idx + 1]] = Yₙ⁺[t_param_idx]
        end
    end

    # TODO Maybe we could return a solution object with the below plus Φ and Φ²
    return (Φₙ_increments, Φₙ²_increments, Yₙ)
end

"""
    analyze(s_model::CallableModel)
    analyze(X::DataFrame, Y::Vector, perms::Matrix)
    analyze(S::CallableModelSample, Y::Vector)

TODO Rename `analyze` to `shapley_effect`?

Dependent factors are handled at sampling time: build the samples with
`CallableModelSample(X1, X2; conditional_sampler=...)`, run the model over `S.samples`, then call
`analyze(S, Y)`.

# Arguments
- `s_model` : SAShE CallableModel
- `S` : SAShE sample
- `Y` : Resulting outputs from `X`
- `X` : Inputs used to run target model
- `perms` : Permutation order

# Returns
Tuple, of Φₙ and Φₙ² (Shapley Effect and variance) or tuple of matrices Φₙ, Φ²ₙ, Yₙ, with:

    - Φₙ : Shapley effects for base samples (size `N`)
    - Φ²ₙ : Variance of Shapley effects used to estimate confidence bounds
    - Yₙ : Model run results the parameters `s_model.X1`
"""
function analyze(s_model::CallableModel)
    n_samples = s_model.n_samples

    res = @showprogress pmap(
        _shapley_effect_iteration,
        repeated(s_model.func, n_samples),
        eachrow(s_model.X1),
        eachrow(s_model.X2),
        eachrow(s_model.permutations),
        eachrow(s_model.Y⁻),
        eachrow(s_model.Y⁺),
        eachrow(s_model.Φ_increments),
        eachrow(s_model.Φ²_increments),
        repeated(s_model.n_samples, n_samples),
    )

    # TODO Return a better object, either a `Solution` or a new version of `CallableModel`
    return hcat([r[1] for r ∈ res]...), hcat([r[2] for r ∈ res]...), [r[3] for r ∈ res]
end
function analyze(S::CallableModelSample, Y::Vector)
    X = S.samples
    return analyze(X, Y, S.permutations)
end
function analyze(X::DataFrame, Y::Vector, perms::Matrix)
    n_var_params = size(X, 2)
    n_base_samples = size(perms, 1)

    check_size = Int64(size(X, 1) / (n_var_params + 1))
    @assert check_size == n_base_samples "Sample sizes do not match!"

    Φₙ_increments = zeros(n_base_samples, n_var_params)
    Φₙ²_increments = zeros(n_base_samples, n_var_params)

    Yₙ⁻ = zeros(n_var_params)
    Yₙ⁺ = zeros(n_var_params)

    # For each sample...
    for n ∈ 1:n_base_samples
        # Calculate the starting index for this base sample
        base_idx = (n - 1) * (n_var_params + 1) + 1

        # Yₙ⁻ is the result for Xₙ, and gets replaced by Yₙ⁺
        # Yₙ⁺ is the result for Xₙ₊₁
        πₙ = perms[n, :]
        Yₙ = Y[base_idx]

        Yₙ⁻ .= 0.0
        Yₙ⁺ .= 0.0
        Yₙ⁻[πₙ[1]] = Yₙ  # Set first value according to permutation

        for param_idx ∈ 1:n_var_params
            eval_idx = base_idx + param_idx
            t_param_idx = πₙ[param_idx]

            Yₙ⁺[t_param_idx] = Y[eval_idx]

            f_diff = (Yₙ⁻[t_param_idx] - Yₙ⁺[t_param_idx])
            f_arg = (Yₙ - Yₙ⁻[t_param_idx] / 2 - Yₙ⁺[t_param_idx] / 2) * f_diff

            Φₙ_increments[n, t_param_idx] = f_arg * (1 / n_base_samples)
            Φₙ²_increments[n, t_param_idx] = f_arg^2 * (1 / n_base_samples)

            if param_idx < n_var_params
                Yₙ⁻[πₙ[param_idx + 1]] = Yₙ⁺[t_param_idx]
            end
        end
    end

    return (Matrix(Φₙ_increments'), Matrix(Φₙ²_increments'))
end
