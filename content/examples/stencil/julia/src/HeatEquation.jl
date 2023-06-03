module HeatEquation

using ProgressMeter
using Plots

include("setup.jl")
include("io.jl")
include("core.jl")

export Field, simulate!, initialize, visualize, average_temperature

end # module
