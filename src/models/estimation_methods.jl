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
[`CallablePickAndFreezeSample`](@ref) — there is no separate estimator argument there, since
the sample's type already determines which method runs.
"""
struct PickAndFreeze <: EstimationMethod end

"""
    DoubleMonteCarlo()

The double (nested) Monte Carlo estimation method of [2], §4.1, Algorithm 1 — estimates
each coalition's cost via a genuinely new inner/outer sampling procedure, rather than
reusing paired samples. For a [`CallableModel`](@ref), this is selected implicitly by
pairing it with a [`CallableDoubleMonteCarloSample`](@ref), built from per-factor
distributions since it draws far more samples than pick-and-freeze's paired-sample table
holds. Pass this explicitly to [`DataModel`](@ref)'s `analyze` to run [1]'s "knn" double
Monte-Carlo estimator (§6.1.1, Eq. 18) instead — see
[`analyze(model::DataModel, n_permutations::Integer, ::DoubleMonteCarlo)`](@ref).
"""
struct DoubleMonteCarlo <: EstimationMethod end
