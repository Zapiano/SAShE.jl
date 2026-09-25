using Statistics: var

"""
    CallableModel(func::Function)

Wraps a callable model for use with [`analyze`](@ref) — the "what you have" container for
the case where inputs are sampled from a known distribution (independent or dependent).
Carries no sample data itself: pair it with a [`PickAndFreezeSample`](@ref) or
[`DoubleMonteCarloSample`](@ref), built independently, to run and analyze `func`.

# Arguments
- `func` : A function that accepts a vector of factor values and returns a scalar.

# Examples
```julia
m = CallableModel(ishigami)
s = PickAndFreezeSample(X1, X2)
Φₙ, Φ²ₙ = analyze(m, s)
```
"""
struct CallableModel
    func::Function
end

function Base.:show(io::IO, m::CallableModel)
    return print(io, "CallableModel(", m.func, ")")
end

"""
    analyze(m::CallableModel, s::PickAndFreezeSample)::Tuple{Matrix{Float64},Matrix{Float64},Vector}
    analyze(m::CallableModel, s::DoubleMonteCarloSample)::Tuple{Matrix{Float64},Matrix{Float64},Vector}

Run `m.func` over every row of `s.samples` (in parallel, via `pmap`), then estimate Shapley
effects. Which estimator runs is determined entirely by `s`'s type — there is no
`estimator=` keyword to choose it explicitly:

- `PickAndFreezeSample` → the pick-and-freeze method of [4] (Algorithm 1).
- `DoubleMonteCarloSample` → [2]'s double Monte Carlo estimator (§4.1, Algorithm 1), which
  estimates each coalition's cost function `c(u) = E[Var[Y | X₋ᵤ]]` directly via nested
  sampling rather than pairing already-known values — chosen by [2] specifically because
  this cost function's estimator is unbiased for any sample size, unlike the alternative
  `Var[E[Y|Xᵤ]]` pick-and-freeze relies on.

# Arguments
- `m` : The model to run, wrapped in a [`CallableModel`](@ref).
- `s` : The sample table to evaluate `m.func` over, wrapped in a [`PickAndFreezeSample`](@ref)
  or a [`DoubleMonteCarloSample`](@ref).

# Returns
Tuple `(Φₙ, Φ²ₙ, Yₙ)`:

    - Φₙ : Per-sample Shapley-effect increments — pass to [`shapley_effects`](@ref) or
      [`confint`](@ref) for final effects and confidence bounds.
    - Φ²ₙ : Their squares, used by the same functions.
    - Yₙ : `m.func` evaluated at every row of `s.samples`, in case you want it (e.g. to
      sanity-check `sum(Φ)` against `var(Yₙ)`).

See the [References](@ref) page for the full citations behind [2] and [4].
"""
function analyze(m::CallableModel, s::PickAndFreezeSample)
    Y = pmap(row -> m.func(collect(row)), eachrow(s.samples))

    X = s.samples
    perms = s.permutations
    n_var_params = size(X, 2)
    n_base_samples = size(perms, 1)

    # `Int64(...)` on a non-integral ratio would throw `InexactError` before the `@assert`
    # that used to follow could report anything, so check the row count directly.
    @assert size(X, 1) == n_base_samples * (n_var_params + 1) "Sample sizes do not match!"

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

    return Matrix(Φₙ_increments'), Matrix(Φₙ²_increments'), Y
end
function analyze(m::CallableModel, s::DoubleMonteCarloSample)
    Y = pmap(row -> m.func(collect(row)), eachrow(s.samples))

    N_V, N_O, N_I = s.N_V, s.N_O, s.N_I
    perms = s.permutations
    n_perms, n_factors = size(perms)

    var_y = var(@view Y[1:N_V])

    rows_per_step = N_O * N_I
    rows_per_permutation = (n_factors - 1) * rows_per_step

    Φₙ_increments = zeros(n_factors, n_perms)
    Φₙ²_increments = zeros(n_factors, n_perms)

    for ℓ ∈ 1:n_perms
        π = @view perms[ℓ, :]
        base = N_V + (ℓ - 1) * rows_per_permutation

        prevW = 0.0
        for j ∈ 1:n_factors
            W = if j == n_factors
                var_y
            else
                step_start = base + (j - 1) * rows_per_step
                Wsum = 0.0
                for l ∈ 1:N_O
                    inner_start = step_start + (l - 1) * N_I
                    Wsum += var(@view Y[(inner_start + 1):(inner_start + N_I)])
                end
                Wsum / N_O
            end

            Δ = W - prevW
            Φₙ_increments[π[j], ℓ] = Δ / n_perms
            Φₙ²_increments[π[j], ℓ] = Δ^2 / n_perms

            prevW = W
        end
    end

    return Φₙ_increments, Φₙ²_increments, Y
end
