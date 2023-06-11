#using Plots
using BenchmarkTools

include("heat.jl")
include("core.jl")


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

# initialize current and previous states to the same state
curr, prev = initialize(ncols, nrows)

# visualize initial field, requires Plots.jl
#visualize(curr, "initial.png")

# simulate temperature evolution for nsteps
simulate!(curr, prev, nsteps)

# visualize final field, requires Plots.jl
#visualize(curr, "final.png")
