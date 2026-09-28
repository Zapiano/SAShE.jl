module SAShE

using Base.Iterators: repeated
using DataFrames, Distributed, ProgressMeter

using DocStringExtensions

include("samplers/sampling_strategies.jl")
include("samplers/callable_samplers.jl")
include("samplers/mix_samplers.jl")
include("models/estimation_methods.jl")
include("models/nearest_neighbour_search.jl")
include("models/callable_model.jl")
include("models/data_model.jl")
include("models/mix_model.jl")
include("shapley_effects.jl")

export CallableModel, CallablePickAndFreezeSample, CallableDoubleMonteCarloSample
export DataModel, MixModel, MixPickAndFreezeSample, MixDoubleMonteCarloSample
export EstimationMethod, PickAndFreeze, DoubleMonteCarlo
export SamplingStrategy, MonteCarloSampling, QuasiMonteCarloSampling
export analyze
export generate_permutations

end
