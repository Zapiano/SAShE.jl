"""
    EstimationMethod

Abstract supertype for `analyze`'s `estimator=` keyword. Concrete subtypes are singleton
strategy structs (e.g. [`PickAndFreeze`](@ref)) selecting how a Shapley effect is estimated,
independent of which container (`CallableModel`, `DataModel`, ...) holds the data.
"""
abstract type EstimationMethod end

"""
    PickAndFreeze()

The pick-and-freeze estimation method — the default and, for now, only `estimator=` value.
Every `analyze` method currently implements this method; passing anything else raises an
error until a second method (e.g. double Monte Carlo) exists.
"""
struct PickAndFreeze <: EstimationMethod end
