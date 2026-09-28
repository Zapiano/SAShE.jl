# `ishigami` comes from test/reference_values.jl.

# Dependency: x3 = x1 + x2, with x1, x2 ~ Uniform(-π, π) independently.
function conditional_sampling(X1_param_idx, X2_param_idx, X1, X2)
    has(set, i) = i ∈ set

    drawn = if has(X2_param_idx, 3)
        x1 = has(X1_param_idx, 1) ? X1[1] : X2[1]
        x2 = has(X1_param_idx, 2) ? X1[2] : X2[2]
        (x1, x2, x1 + x2)
    elseif has(X2_param_idx, 1) && has(X2_param_idx, 2)
        sum_value = X1[3]                                # x1 + x2, frozen
        lb, ub = max(-π, sum_value - π), min(π, sum_value + π)
        x1 = lb + ((X2[1] + π) / (2π)) * (ub - lb)       # remap X2[1] ~ U(-π,π) to U(lb,ub)
        (x1, sum_value - x1, sum_value)
    else
        # x3 and one addend frozen ⇒ the other is forced to its base value
        (X1[3] - X1[2], X1[3] - X1[1], X1[3])
    end

    return [drawn[i] for i in X2_param_idx]
end

@testset "Conditional sampling: manual build vs conditional_sampler kwarg" begin
    Random.seed!(0987)

    factor_names = [:x1, :x2, :x3]
    n_samples, n_factors = 2000, 3
    du = Uniform(-π, π)

    # Base samples: x1, x2 drawn independently; x3 = x1 + x2.
    function constrained_samples()
        m = zeros(n_samples, n_factors)
        m[:, 1:2] .= hcat([rand(du, 2) for _ ∈ 1:n_samples]...)'
        m[:, 3] .= m[:, 1] .+ m[:, 2]
        return DataFrame(m, factor_names)
    end

    A, B = constrained_samples(), constrained_samples()
    permutations = SAShE.generate_permutations(n_samples, n_factors)

    # Way 1 - the package builds Z and conditions it via the kwarg.
    S_auto = CallablePickAndFreezeSample(A, B, permutations; conditional_sampler=conditional_sampling)

    # Way 2 - user builds the plain pick-freeze sample (no kwarg), then conditions every
    # non-base row by hand (the workflow a user drives themselves, public API only).
    S_manual = CallablePickAndFreezeSample(A, B, permutations)
    block_size = n_factors + 1
    for i ∈ 1:n_samples, pos ∈ 1:n_factors
        πₙ = permutations[i, :]
        row = (i - 1) * block_size + 1 + pos
        X2_param_idx = πₙ[1:pos]
        X1_param_idx = πₙ[(pos + 1):end]
        values = conditional_sampling(
            X1_param_idx, X2_param_idx, collect(A[i, :]), collect(B[i, :])
        )
        for (col, value) ∈ zip(X2_param_idx, values)
            S_manual.samples[row, col] = value
        end
    end

    # Both routes must produce a bit-identical sample matrix ...
    @test Matrix(S_auto.samples) == Matrix(S_manual.samples)

    # ... every generated row respects the dependency x3 = x1 + x2 ...
    @test all(isapprox.(S_auto.samples.x3, S_auto.samples.x1 .+ S_auto.samples.x2; atol=1e-9))

    # ... and the downstream Shapley-effect analysis is identical.
    m = SAShE.CallableModel(ishigami)
    Φₙ_auto, Φ²ₙ_auto, _ = SAShE.analyze(m, S_auto)
    Φₙ_manual, Φ²ₙ_manual, _ = SAShE.analyze(m, S_manual)
    @test Φₙ_auto == Φₙ_manual
    @test Φ²ₙ_auto == Φ²ₙ_manual
end
