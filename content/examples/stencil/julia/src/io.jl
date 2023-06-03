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