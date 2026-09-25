"""
    EstimationMethod

Abstract supertype for the estimation-method argument some `analyze` methods dispatch on
(e.g. [`DataModel`](@ref)'s). Concrete subtypes are singleton strategy structs (e.g.
[`PickAndFreeze`](@ref)) selecting how a Shapley effect is estimated, independent of which
container (`CallableModel`, `DataModel`, ...) holds the data.
"""
abstract type EstimationMethod end

"""
    PickAndFreeze()

The pick-and-freeze estimation method — pass explicitly to [`DataModel`](@ref)'s `analyze`
to reuse values already present in the dataset (no new function evaluations). For a
[`CallableModel`](@ref), pick-and-freeze is selected implicitly by pairing it with a
[`PickAndFreezeSample`](@ref) — there is no separate estimator argument there, since the
sample's type already determines which method runs.
"""
struct PickAndFreeze <: EstimationMethod end

"""
    DoubleMonteCarlo()

The double (nested) Monte Carlo estimation method of [2], §4.1, Algorithm 1 — estimates
each coalition's cost via a genuinely new inner/outer sampling procedure, rather than
reusing paired samples. For a [`CallableModel`](@ref), this is selected implicitly by
pairing it with a [`DoubleMonteCarloSample`](@ref), built from per-factor distributions
since it draws far more samples than pick-and-freeze's paired-sample table holds.

!!! note "No `analyze` method dispatches on this singleton yet"
    Because the `CallableModel` path selects double Monte Carlo from the *sample's* type, and
    [`DataModel`](@ref) only implements [`PickAndFreeze`](@ref) so far, nothing currently
    accepts this value — it is a placeholder for `analyze(::DataModel, n, ::DoubleMonteCarlo)`.
    Passing it to today's `analyze` raises a `MethodError`.

See the [References](@ref) page for the full citation behind [2].
"""
struct DoubleMonteCarlo <: EstimationMethod end
