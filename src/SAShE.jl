module SAShE

using Base.Iterators: repeated
using DataFrames, Distributed, ProgressMeter

using DocStringExtensions

include("model.jl")
include("samples.jl")
include("shapley_effect.jl")

export SAShEModel, SAShESample
export analyze

end
