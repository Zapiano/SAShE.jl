"""
    s_model (X1::DataFrame, X2::DataFrame, Y::Vector, Y⁻::Matrix, Y⁺::Matrix)
    s_model (X1::DataFrame, X2::DataFrame)

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
struct SAShEModel
    func::Function
    X1::DataFrame                       # What the paper calls x
    X2::DataFrame                       # What the paper calls y
    Y::Vector                           # What the paper calls F
    Y⁻::Matrix                          # What the paper calls F⁻
    Y⁺::Matrix                          # What the paper calls F⁺
    permutations::Matrix{Int64}                # What the paper calls π
    Φ_increments::Matrix{Float64}
    Φ²_increments::Matrix{Float64}
    n_samples::Int64

    # TODO Rename SAShEModel to Model?
    function SAShEModel(func::Function, X1::DataFrame, X2::DataFrame)
        # Validate inputs
        _validate_sashe_model(X1, X2)

        n_samples, n_factors = size(X1)
        _Y::Vector = zeros(Float64, n_samples)
        # TODO Switch rows and cols so each col is a sample to improve performance
        _Y⁻::Matrix = zeros(Float64, n_samples, n_factors)
        _Y⁺::Matrix = zeros(Float64, n_samples, n_factors)
        _permutations = generate_permutations(n_samples, n_factors)
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
            n_samples,
        )
    end
end

function _validate_sashe_model(X1::DataFrame, X2::DataFrame)
    size_error_msg = "`samples_X1` and `samples_X2` must have the same size"
    factor_names_error_msg = "`samples_X1` and `samples_X2` must have the same factors"
    errors::Vector{String} = []
    (size(X1) == size(X2)) || push!(errors, size_error_msg)
    names(X1) == names(X2) || push!(errors, factor_names_error_msg)
    return !isempty(errors) ? error(join(errors, "\n")) : nothing
end

function Base.:show(io::IO, p::SAShEModel)
    println(p.func)
    return println("n_samples: ", p.n_samples)
end
