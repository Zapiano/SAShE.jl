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
