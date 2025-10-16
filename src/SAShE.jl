module SAShE

using Base.Iterators: repeated
using DataFrames, Distributed, ProgressMeter

using DocStringExtensions

include("samples.jl")
include("shapley_effect.jl")

export problem, SAShESample
export solve, analyze

end
