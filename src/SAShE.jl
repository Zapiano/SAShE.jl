module SAShE

using Base.Iterators: repeated
using DataFrames, Distributed, ProgressMeter

using DocStringExtensions

include("estimation_methods.jl")
include("sampling_strategies.jl")
include("callable_samplers.jl")
include("callable_model.jl")
include("nearest_neighbour_search.jl")
include("data_model.jl")
include("mix_samplers.jl")
include("mix_model.jl")
include("shapley_effects.jl")

export CallableModel, CallablePickAndFreezeSample, CallableDoubleMonteCarloSample
export DataModel, MixModel, MixPickAndFreezeSample, MixDoubleMonteCarloSample
export EstimationMethod, PickAndFreeze, DoubleMonteCarlo
export SamplingStrategy, MonteCarloSampling, QuasiMonteCarloSampling
export analyze
export generate_permutations

end
