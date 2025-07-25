
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
    shapley_effects(Φₙ, Φ²ₙ)

# Arguments
- `Φₙ` : Shapley effects for each sample (rows) and factor (cols)
- `Φ²ₙ` : Variance of the Shapley effects for each sample (rows) and factor (cols)

# Returns
Returns a tuple with the total Shapley effects and their respective std.
"""
function shapley_effects(Φₙ)::Vector{Float64}
    return dropdims(sum(Φₙ, dims=2), dims=2)
end
function shapley_effects(Φₙ, Φ²ₙ)
    Φ = shapley_effects(Φₙ)
    moe = margin_of_error(Φₙ, Φ²ₙ)
    Φ, Φ .- moe, Φ .+ moe
end

function _shapley_effect_iteration(
    func::Function,
    X1ₙ::DataFrameRow,
    X2ₙ::DataFrameRow,
    πₙ::Vector{Int64},          # A row of the permutation matrix.
    Yₙ::Float64,
    Yₙ⁻::Vector{Float64},
    Yₙ⁺::Vector{Float64},
    Φₙ_increments::Vector{Float64},
    Φₙ²_increments::Vector{Float64},
    factor_names::Vector{String},
    n_samples::Int64,
)
    Zₙ = deepcopy(X1ₙ)
    Yₙ = func(collect(Zₙ)) #sum(@view ADRIA.run_model(dom, X1ₙ).raw[end, :, :])
    Yₙ⁻[πₙ[1]] = Yₙ

    n_var_params = length(factor_names)
    t_param_idx::Int64 = 1
    for param_idx ∈ 1:n_var_params
        # Target param index is different from param_idx.
        t_param_idx = πₙ[param_idx]

        param_names_X2 = factor_names[πₙ[1:(param_idx)]]
        param_names_X1 = factor_names[πₙ[(param_idx+1):end]]

        Zₙ[param_names_X2] = X2ₙ[param_names_X2]
        Zₙ[param_names_X1] = X1ₙ[param_names_X1]

        Yₙ⁺[t_param_idx] = func(collect(Zₙ))

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

# TODO Rename `solve` to `shapley_effect` ?
function solve(_problem::Problem)
    n_samples = _problem.n_samples

    factor_names = names(_problem.X1)

    res = @showprogress pmap(
        _shapley_effect_iteration,
        fill(_problem.func, n_samples),
        [_problem.X1[i, :] for i in 1:n_samples],
        [_problem.X2[i, :] for i in 1:n_samples],
        [_problem.permutations[i, :] for i in 1:n_samples],
        _problem.Y,
        [_problem.Y⁻[i, :] for i in 1:n_samples],
        [_problem.Y⁺[i, :] for i in 1:n_samples],
        [_problem.Φ_increments[i, :] for i in 1:n_samples],
        [_problem.Φ²_increments[i, :] for i in 1:n_samples],
        fill(factor_names, n_samples),
        fill(_problem.n_samples, n_samples)
    )

    # TODO Return a better object, either a `Solution` or a new version of `Problem`

    return hcat([r[1] for r in res]...), hcat([r[2] for r in res]...), [r[3] for r in res]
end

function Base.:show(io::IO, p::Problem)
    println(p.func)
    println("n_samples: ", p.n_samples)
end
