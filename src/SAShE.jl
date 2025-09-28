module SAShE

using Base.Iterators: repeated
using DataFrames, Distributed, ProgressMeter

using DocStringExtensions

include("shapley_effect.jl")
include("samples.jl")

export problem, SAShESample
export solve

end
