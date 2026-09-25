"""
    shapley_effects(Φₙ)::Vector{Float64}

Sum each factor's per-sample increments into its Shapley effect.

# Arguments
- `Φₙ` : Shapley-effect increments, `d × N` — rows are factors, columns are samples (or
  permutations), as returned by `analyze`.

# Returns
Vector of length `d`, the estimated Shapley effects.
"""
function shapley_effects(Φₙ::AbstractArray{Float64, 2})::Vector{Float64}
    return dropdims(sum(Φₙ; dims=2); dims=2)
end

"""
    margin_of_error(Φₙ, Φ²ₙ)::Vector{Float64}

The 95% margin of error (`1.96 × standard error`) of each factor's Shapley effect, computed
from the sample variance of its per-sample increments — no bootstrapping needed.

The formula treats the columns of `Φₙ` as independent draws, and `Var[Y]` (where an estimator
uses one) as known. See the [References](@ref) page for the conditions each estimator
establishes.

# Arguments
- `Φₙ` : Shapley-effect increments, `d × N` — rows are factors, columns are samples (or
  permutations), as returned by `analyze`.
- `Φ²ₙ` : The same increments, squared elementwise (not their variance) — also returned by
  `analyze`, alongside `Φₙ`.

# Returns
Vector of length `d`: each factor's margin of error, in the same units as its Shapley
effect.
"""
function margin_of_error(Φₙ::Matrix{Float64}, Φ²ₙ::Matrix{Float64})::Vector{Float64}
    n_samples = size(Φₙ, 2)
    n_samples > 1 || throw(
        ArgumentError(
            "Need at least 2 samples (columns of Φₙ) to estimate a standard error, got " *
            "$n_samples.",
        ),
    )
    # E[Φ²] == sum(Φ²ₙ, dims=2) and (E[Φ])² == (sum(Φₙ, dims=2)).^2
    stds = sqrt.((shapley_effects(Φ²ₙ) .- (shapley_effects(Φₙ) .^ 2)) ./ (n_samples - 1))
    return 1.96 .* stds
end

"""
    confint(Φₙ, Φ²ₙ)::Tuple{Vector{Float64},Vector{Float64}}

95% confidence bounds for each factor's Shapley effect: `shapley_effects(Φₙ)` plus and minus
[`margin_of_error`](@ref).

# Arguments
- `Φₙ` : Shapley-effect increments, `d × N` — rows are factors, columns are samples (or
  permutations), as returned by `analyze`.
- `Φ²ₙ` : The same increments, squared elementwise (not their variance) — also returned by
  `analyze`, alongside `Φₙ`.

# Returns
Tuple `(Φlb, Φub)`, each a vector of length `d`: the lower and upper 95% confidence bound
for every factor's Shapley effect.
"""
function confint(Φₙ::Matrix{Float64}, Φ²ₙ::Matrix{Float64})
    Φ = shapley_effects(Φₙ)
    moe = margin_of_error(Φₙ, Φ²ₙ)
    return Φ .- moe, Φ .+ moe
end

"""
    shapley_effects(Φₙ, Φ²ₙ)::NTuple{3,Vector{Float64}}

Convenience form combining [`shapley_effects(Φₙ)`](@ref) and [`confint`](@ref) in one call.

# Arguments
- `Φₙ` : Shapley-effect increments, `d × N` — rows are factors, columns are samples (or
  permutations), as returned by `analyze`.
- `Φ²ₙ` : The same increments, squared elementwise (not their variance) — also returned by
  `analyze`, alongside `Φₙ`.

# Returns
Tuple `(Φ, Φlb, Φub)`, each a vector of length `d`: the Shapley effect, and its lower and
upper 95% confidence bound, for every factor.
"""
function shapley_effects(
    Φₙ::AbstractArray{Float64, 2}, Φ²ₙ::AbstractArray{Float64, 2}
)::NTuple{3, Vector{Float64}}
    Φ = shapley_effects(Φₙ)
    moe = margin_of_error(Φₙ, Φ²ₙ)
    return Φ, Φ .- moe, Φ .+ moe
end
