using Test
using SAShE
using DataFrames
using Distributions
using Random
using Statistics
import QuasiMonteCarlo as QMC

include("reference_values.jl")  # shared `ishigami` + exact values; must come first
include("ishigami.jl")
include("conditional.jl")
include("nearest_neighbours.jl")
include("nearest_neighbour_shapley_effects.jl")
include("double_monte_carlo.jl")
include("container_types.jl")
include("estimation_methods.jl")
include("sampling_strategies.jl")
include("mix_model.jl")
