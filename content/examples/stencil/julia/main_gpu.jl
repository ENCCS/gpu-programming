#using Plots
using BenchmarkTools
using AMDGPU

include("heat.jl")
include("core_gpu.jl")


"""
    visualize(curr::Field, filename=:none)

Create a heatmap of a temperature field. Optionally write png file. 
"""    
function visualize(curr::Field, filename=:none)
    background_color = :white
    plot = heatmap(
        curr.data,
        colorbar_title = "Temperature (C)",
        background_color = background_color
    )

    if filename != :none
        savefig(filename)
    else
        display(plot)
    end
end


ncols, nrows = 2048, 2048
nsteps = 500

# initialize data on CPU
curr, prev = initialize(ncols, nrows, ROCArray)
# initialize data on CPU
#curr, prev = initialize(ncols, nrows)

# visualize initial field, requires Plots.jl
#visualize(curr, "initial.png")

# simulate temperature evolution for nsteps
@btime simulate!(curr, prev, nsteps)

# visualize final field, requires Plots.jl
#visualize(curr, "final.png")
