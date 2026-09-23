module SAShE

using Base.Iterators: repeated
using DataFrames, Distributed, ProgressMeter

using DocStringExtensions

include("estimators.jl")
include("callable_model.jl")
include("nearest_neighbour_search.jl")
include("data_model.jl")
include("mix_model.jl")
include("shapley_effects.jl")

export CallableModel, CallableModelSample, DataModel, MixModel
export EstimationMethod, PickAndFreeze
export analyze
export generate_permutations

end
