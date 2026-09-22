module SAShE

using Base.Iterators: repeated
using DataFrames, Distributed, ProgressMeter

using DocStringExtensions

include("model.jl")
include("samples.jl")
include("shapley_effect.jl")
include("nearest_neighbours.jl")

export SAShEModel, SAShESample
export analyze
export generate_permutations

end
